require(SHAP_PATH.is_file(), f'Valeurs de Shapley absentes ({SHAP_PATH.name}) : activer protected_ablation')
shap_frame = pd.read_parquet(SHAP_PATH)
shap_values = shap_frame.drop(columns=['id', 'expected_value'])
require(shap_values.columns.equals(full_train.columns) and len(shap_values) == SHAP_ROWS,
        'Valeurs de Shapley incompatibles avec les variables')
shap_rows = pd.Index(train_ids).get_indexer(shap_frame['id'].astype(str))
require((shap_rows >= 0).all() and (fold_assignment[shap_rows] == SHAP_FOLD).all(), 'Lignes de Shapley hors du pli attendu')
shap_features = full_train.iloc[shap_rows].reset_index(drop=True)
importance = shap_values.abs().mean().sort_values(ascending=False)


def family(column):
    if column in PROTECTED_COLUMNS:
        return 'Variables protégées (origine, sexe, âge)'
    if 'ltv' in column or column in {'loan_to_value_ratio', 'property_value'}:
        return 'LTV et valeur du bien'
    if column in {'debt_to_income_ratio', 'dti_lower', 'dti_upper', 'income',
                  'income_to_area', 'loan_to_income', 'value_to_income'}:
        return 'Endettement et revenu'
    if column in {'state_code', 'derived_msa-md', 'county_code', 'ffiec_msa_md_median_family_income'} or column.startswith('tract_'):
        return 'Géographie et secteur'
    if column in {'aus-1', 'submission_of_application', 'initially_payable_to_institution', 'preapproval',
                  'applicant_credit_score_type', 'co-applicant_credit_score_type'}:
        return 'Instruction du dossier (AUS, score, dépôt)'
    if column.startswith('missing_'):
        return 'Profil de valeurs manquantes'
    return 'Produit, montant et durée'


families = importance.groupby(importance.index.map(family)).agg(['size', 'sum'])
families.columns = ['Colonnes', 'Somme des |SHAP| moyens']
families['Part'] = families['Somme des |SHAP| moyens'] / importance.sum()
display_named(families.sort_values('Part', ascending=False).round(3))
print(f'Les 5 premières variables : {importance.head(5).sum() / importance.sum():.0%} du total ; '
      f'les 10 premières : {importance.head(10).sum() / importance.sum():.0%}.')

top = importance.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6.5))
ax.barh(range(len(top)), top.to_numpy(),
        color=['#c44e52' if c in PROTECTED_COLUMNS else '#4c72b0' for c in top.index])
ax.set_yticks(range(len(top)), top.index)
ax.set_xlabel('|SHAP| moyen (log-odds) ; rouge = variable protégée')
ax.set_title(f'20 variables les plus influentes (pli {SHAP_FOLD}, {SHAP_ROWS} lignes de validation)')
plt.tight_layout()
plt.show()

numeric_top = [c for c in importance.index if c not in full_cats][:3]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, column in zip(axes, numeric_top):
    values = shap_features[column]
    low, high = values.quantile([0.01, 0.99])
    keep = values.between(low, high)
    ax.scatter(values[keep], shap_values.loc[keep, column], s=4, alpha=0.35, color='#4c72b0')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel(column)
    ax.set_ylabel('SHAP (log-odds)')
fig.suptitle('Effet des trois variables numériques les plus influentes (1er au 99e centile)')
plt.tight_layout()
plt.show()
