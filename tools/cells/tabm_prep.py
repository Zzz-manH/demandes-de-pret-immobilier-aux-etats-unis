nn_train, nn_test, nn_cats = variant_frames(ENRICHED_FEATURE_BLOCKS)
nn_nums = [c for c in nn_train.columns if c not in nn_cats]
print(f'Variables : {nn_train.shape[1]} colonnes = {len(nn_nums)} numériques + {len(nn_cats)} catégorielles')


class FoldPreprocessor:
    """Ajustement sur le train externe uniquement : normalisation quantile gaussienne, PLE, indicateurs de valeurs manquantes et vocabulaire catégoriel."""

    def __init__(self, bins, rare_min_count):
        self.bins = bins
        self.rare_min_count = rare_min_count

    def fit(self, frame):
        num = frame[nn_nums].to_numpy(dtype='float64')
        self.quantile = QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                                            subsample=10**9, random_state=SEED)
        self.quantile.fit(num)                      # sklearn ignore les NaN
        self.edges = []
        for j in range(num.shape[1]):
            values = num[:, j][~np.isnan(num[:, j])]
            edges = np.unique(np.quantile(values, np.linspace(0, 1, self.bins + 1)))
            self.edges.append(edges if len(edges) >= 2 else np.array([values.min(), values.min() + 1]))
        self.has_missing = np.isnan(num).any(axis=0)
        self.vocab = []
        for column in nn_cats:
            counts = frame[column].value_counts()
            kept = counts.index[counts >= self.rare_min_count]
            self.vocab.append({value: i + 1 for i, value in enumerate(kept)})   # 0 = inconnu / rare
        self.cardinalities = [len(v) + 1 for v in self.vocab]
        return self

    def transform(self, frame):
        num = frame[nn_nums].to_numpy(dtype='float64')
        missing = np.isnan(num)
        normal = np.nan_to_num(np.clip(self.quantile.transform(num), -5, 5), nan=0.0)
        ple_parts = []
        for j, edges in enumerate(self.edges):
            x = num[:, j:j + 1]
            left, right = edges[:-1][None, :], edges[1:][None, :]
            part = np.clip((x - left) / np.maximum(right - left, 1e-12), 0, 1)
            part = np.nan_to_num(part, nan=0.0)
            ple_parts.append(part)
        dense = np.concatenate([normal] + ple_parts + [missing[:, self.has_missing]], axis=1)
        cats = np.stack([frame[c].map(v).fillna(0).to_numpy(dtype='int64')
                         for c, v in zip(nn_cats, self.vocab)], axis=1)
        return dense.astype('float32'), cats


class TabMMini(nn.Module):
    """TabM-mini : k membres partagent un MLP, avec chacun un vecteur de facteurs ±1 en entrée. La prédiction moyenne leurs probabilités."""

    def __init__(self, d_dense, cardinalities, k, width, depth, dropout, cat_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, cat_dim) for c in cardinalities])
        d_in = d_dense + cat_dim * len(cardinalities)
        self.adapter = nn.Parameter(torch.randint(0, 2, (k, d_in)).float() * 2 - 1)
        layers = []
        for i in range(depth):
            layers += [nn.Linear(d_in if i == 0 else width, width), nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.head_weight = nn.Parameter(torch.randn(k, width) / width ** 0.5)
        self.head_bias = nn.Parameter(torch.zeros(k))

    def forward(self, dense, cats):
        parts = [dense] + [emb(cats[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(parts, dim=1)                       # (B, D)
        x = x[:, None, :] * self.adapter[None]            # (B, k, D)
        h = self.backbone(x)                              # (B, k, width)
        return (h * self.head_weight).sum(-1) + self.head_bias   # (B, k) logits


def predict(model, dense, cats, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(dense), batch):
            d = torch.from_numpy(dense[start:start + batch]).to(DEVICE)
            c = torch.from_numpy(cats[start:start + batch]).to(DEVICE)
            out.append(torch.sigmoid(model(d, c)).mean(1).cpu().numpy())
    return np.concatenate(out).astype('float64')


def train_fold(fold):
    cfg = NN_CONFIG
    fit_idx = np.flatnonzero(fold_assignment != fold)
    valid_idx = np.flatnonzero(fold_assignment == fold)
    inner_fit, inner_stop = train_test_split(
        fit_idx, test_size=cfg['inner_stop_fraction'], random_state=SEED, stratify=y.iloc[fit_idx])
    prep = FoldPreprocessor(cfg['ple_bins'], cfg['rare_min_count']).fit(nn_train.iloc[fit_idx])
    data = {name: prep.transform(frame) for name, frame in [
        ('fit', nn_train.iloc[inner_fit]), ('stop', nn_train.iloc[inner_stop]),
        ('valid', nn_train.iloc[valid_idx]), ('test', nn_test)]}
    y_fit = torch.from_numpy(y.iloc[inner_fit].to_numpy(dtype='float32')).to(DEVICE)
    y_stop = y.iloc[inner_stop].to_numpy()

    torch.manual_seed(SEED + fold)
    model = TabMMini(data['fit'][0].shape[1], prep.cardinalities, cfg['k'], cfg['width'],
                     cfg['depth'], cfg['dropout'], cfg['cat_dim']).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    dense_fit = torch.from_numpy(data['fit'][0]).to(DEVICE)
    cats_fit = torch.from_numpy(data['fit'][1]).to(DEVICE)
    best_auc, best_state, best_epoch, waited = -1.0, None, 0, 0
    started = time.perf_counter()
    for epoch in range(1, cfg['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(dense_fit), device=DEVICE)
        for start in range(0, len(order), cfg['batch_size']):
            batch = order[start:start + cfg['batch_size']]
            logits = model(dense_fit[batch], cats_fit[batch])
            loss = loss_fn(logits, y_fit[batch][:, None].expand_as(logits))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        stop_auc = roc_auc_score(y_stop, predict(model, *data['stop']))
        if stop_auc > best_auc:
            best_auc, best_epoch, waited = stop_auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        print(f'  fold {fold} epoch {epoch:2d}: AUC du jeu interne d’arrêt anticipé={stop_auc:.5f}'
              f'{"  *" if waited == 0 else ""}', flush=True)
        if waited >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    valid_prob = predict(model, *data['valid'])
    test_prob = predict(model, *data['test'])
    require(np.isfinite(valid_prob).all() and np.isfinite(test_prob).all(), 'Les prédictions du réseau contiennent des valeurs non finies')
    actual = y.iloc[valid_idx]
    stem = OUTPUT_DIR / f'TabMMini__fold{fold}'
    pd.DataFrame({'id': train_ids.iloc[valid_idx].to_numpy(), 'accepted': actual.to_numpy(),
                  'prob': valid_prob}).to_csv(f'{stem}__valid.csv', index=False, float_format='%.12f')
    pd.DataFrame({'id': test_ids.to_numpy(), 'accepted': test_prob}).to_csv(
        f'{stem}__test.csv', index=False, float_format='%.12f')
    result = {
        'group': 'TabMMini', 'fold': fold, 'config': cfg, 'blocks': ENRICHED_FEATURE_BLOCKS,
        'dense_dim': int(data['fit'][0].shape[1]), 'categorical_columns': len(nn_cats),
        'best_epoch': best_epoch, 'inner_stop_auc': best_auc,
        'valid_auc': roc_auc_score(actual, valid_prob),
        'valid_logloss': log_loss(actual, valid_prob, labels=[0, 1]),
        'seconds': time.perf_counter() - started, 'device': DEVICE,
        'torch': torch.__version__, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(f'{stem}.json'), result)
    del model, dense_fit, cats_fit
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    return result
