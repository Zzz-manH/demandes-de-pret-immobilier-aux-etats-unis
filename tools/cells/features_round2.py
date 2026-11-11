def read_raw_ltv(path):
    text = pd.read_csv(path, usecols=['loan_to_value_ratio'], dtype=str,
                       keep_default_na=False)['loan_to_value_ratio']
    return text


raw_ltv = pd.concat([read_raw_ltv(PREPARED_DIR / 'train_features.csv'),
                     read_raw_ltv(PREPARED_DIR / 'test_features.csv')], ignore_index=True)
both = pd.concat([X_base, X_test_base], ignore_index=True)
require(len(raw_ltv) == len(both), 'Nombre de lignes du LTV texte incorrect')
require(pd.to_numeric(raw_ltv.replace('', np.nan)).equals(both['loan_to_value_ratio'])
        or np.allclose(pd.to_numeric(raw_ltv.replace('', np.nan)), both['loan_to_value_ratio'],
                       equal_nan=True), 'LTV texte et numérique incohérents')

decimals = raw_ltv.str.split('.').str[1].fillna('')
precision = np.select(
    [raw_ltv.eq(''), decimals.isin(['', '0']), decimals.str.len().eq(1),
     decimals.str.len().eq(2), decimals.str.len().eq(3)],
    ['missing', 'integer', 'dec1', 'dec2', 'dec3'], default='dec4plus',
)
ltv = both['loan_to_value_ratio']
band = np.select(
    [ltv.isna(), ltv < 80, ltv == 80, ltv < 95, ltv < 97, ltv < 100, ltv == 100],
    ['missing', 'lt80', 'eq80', '80_95', '95_97', '97_100', 'eq100'], default='gt100',
)
PROGRAM_MAX_LTV = {'1': 97.0, '2': 96.5, '3': 100.0, '4': 100.0}   # Conventionnel / FHA / VA / RHS
unknown_types = set(both['loan_type']) - set(PROGRAM_MAX_LTV)
require(not unknown_types, f'Valeurs loan_type inconnues : {sorted(unknown_types)}')
implied_value = safe_ratio(both['loan_amount'] * 100, ltv)
term = both['loan_term']

round2 = {
    'precision': pd.DataFrame({'ltv_precision': precision}),
    'thresholds': pd.DataFrame({
        'ltv_band': band,
        'ltv_over_program': ltv - both['loan_type'].map(PROGRAM_MAX_LTV).astype('float64'),
    }),
    'implied': pd.DataFrame({
        'implied_property_value': implied_value,
        'implied_value_ratio': safe_ratio(implied_value, both['property_value']),
    }),
    'term': pd.DataFrame({
        'term_is_360': term.eq(360).astype('float64').where(term.notna()),
        'term_years': (term / 12).round(1),
    }),
}
n_train = len(X_base)
for name, frame in round2.items():
    require(len(frame) == len(both), f'{name} : nombre de lignes incorrect')
    BLOCKS[name] = (frame.iloc[:n_train].reset_index(drop=True),
                    frame.iloc[n_train:].reset_index(drop=True))
NEW_CATEGORICAL |= {'ltv_precision', 'ltv_band'}

train_labels_check = y.to_numpy()
for column, frame in [('ltv_precision', round2['precision']), ('ltv_band', round2['thresholds'])]:
    table = pd.crosstab(frame[column].iloc[:n_train], train_labels_check, normalize='index')
    counts = frame[column].iloc[:n_train].value_counts()
    display_named(table.assign(rows=counts).round(3))
for name in ['implied', 'term']:
    display_named(BLOCKS[name][0].describe().T.round(3))
del both
