both = pd.concat([X_base, X_test_base], ignore_index=True)
ltv = both['loan_to_value_ratio']
loan, value = both['loan_amount'], both['property_value']

# HMDA 公开数据中贷款额与房价都是 1 万美元区间的中点（训练集全部以 5000 结尾），
# 所以由二者算出的 LTV 只能确定一个区间；申报 LTV 落在区间外，说明用的房价口径不同。
require(((loan % 10000) == 5000).all(), '贷款额不全是 1 万区间中点，区间假设不成立')
ltv_min = 100 * safe_ratio(loan - 5000, value + 5000)
ltv_max = 100 * safe_ratio(loan + 5000, value - 5000)
outside = np.where(ltv.isna() | ltv_min.isna(), np.nan,
                   np.where(ltv < ltv_min, ltv - ltv_min, np.where(ltv > ltv_max, ltv - ltv_max, 0.0)))
consistency = np.select([ltv.isna() | ltv_min.isna(), outside == 0, outside < 0],
                        ['unknown', 'inside', 'below'], default='above')

PRODUCT_FLAGS = ['business_or_commercial_purpose', 'open-end_line_of_credit', 'reverse_mortgage',
                 'interest_only_payment', 'balloon_payment', 'negative_amortization',
                 'other_nonamortizing_features']
purpose = both[PRODUCT_FLAGS].astype(str).agg('|'.join, axis=1)

dti_lower = add_dti_bounds(both)['dti_lower']
risk = pd.DataFrame({
    'high_ltv': ltv > 95,
    'high_dti': dti_lower >= 45,                 # 下界 ≥ 45 才确定为高 DTI
    'no_aus': both['aus-1'].eq('6'),             # aus-1 = 6：未使用自动审批系统
    'not_owner_occupied': both['occupancy_type'].ne('1'),
    'ltv_missing': ltv.isna(),
    'second_lien': both['lien_status'].eq('2'),
})
tract_median_income = both['ffiec_msa_md_median_family_income'] * both['tract_to_msa_income_percentage'] / 100

round3 = {
    'purpose': pd.DataFrame({'product_flags': 'p' + purpose}),
    'ltv_consistency': pd.DataFrame({'ltv_consistency': consistency,
                                     'ltv_outside_interval': outside}),
    'risk_layers': pd.DataFrame({'risk_layer_count': risk.sum(axis=1).astype('float64'),
                                 'risk_layer_pattern': 'r' + risk.astype(int).astype(str).agg(''.join, axis=1)}),
    'tract_income': pd.DataFrame({'income_to_tract': safe_ratio(both['income'] * 1000, tract_median_income)}),
}
n_train = len(X_base)
for name, frame in round3.items():
    require(len(frame) == len(both), f'{name} 行数不符')
    BLOCKS[name] = (frame.iloc[:n_train].reset_index(drop=True),
                    frame.iloc[n_train:].reset_index(drop=True))
NEW_CATEGORICAL |= {'product_flags', 'ltv_consistency', 'risk_layer_pattern'}

for column, block in [('product_flags', 'purpose'), ('ltv_consistency', 'ltv_consistency'),
                      ('risk_layer_count', 'risk_layers')]:
    values = BLOCKS[block][0][column]
    table = pd.crosstab(values, y.to_numpy(), normalize='index').assign(rows=values.value_counts())
    display(table.sort_values('rows', ascending=False).head(10).round(3))
display(BLOCKS['tract_income'][0].describe().T.round(3))
del both
