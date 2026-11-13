without_oof = read_oof(OUTPUT_DIR, WITHOUT_PROTECTED)
require(without_oof['id'].equals(reference_oof['id']), 'ID des deux variantes désalignés')
groups_frame = X_base.assign(id=train_ids.astype('int64').to_numpy()).merge(
    reference_oof[['id', 'accepted', 'prob']], on='id', validate='one_to_one').merge(
    without_oof[['id', 'prob']].rename(columns={'prob': 'prob_sans'}), on='id', validate='one_to_one')
GROUP_COLUMNS = {'derived_race': 'Origine', 'derived_ethnicity': 'Ethnicité',
                 'derived_sex': 'Sexe', 'applicant_age': 'Âge de l’emprunteur'}
MIN_GROUP_ROWS = 500


def group_table(column):
    rows = {}
    for value, part in groups_frame.groupby(column):
        if len(part) < MIN_GROUP_ROWS:
            continue
        rows[value] = {
            'Lignes': len(part), 'Taux observé': part['accepted'].mean(),
            'Probabilité moyenne': part['prob'].mean(),
            'Écart': part['accepted'].mean() - part['prob'].mean(),
            'AUC': roc_auc_score(part['accepted'], part['prob']),
            'Probabilité moyenne, sans': part['prob_sans'].mean(),
            'AUC, sans': roc_auc_score(part['accepted'], part['prob_sans']),
        }
    return pd.DataFrame(rows).T.sort_values('Lignes', ascending=False).astype({'Lignes': int})


GROUP_TABLES = {column: group_table(column) for column in GROUP_COLUMNS}
for column, label in GROUP_COLUMNS.items():
    print(f'\n{label} ({column}), groupes d’au moins {MIN_GROUP_ROWS} lignes')
    display_named(GROUP_TABLES[column][['Lignes', 'Taux observé', 'Probabilité moyenne', 'Écart', 'AUC']].round(4))
