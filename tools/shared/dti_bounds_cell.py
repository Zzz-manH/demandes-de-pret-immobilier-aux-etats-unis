DTI_BOUNDS = {
    **{str(value): (float(value), float(value)) for value in range(36, 50)},
    '20%-<30%': (20.0, 30.0),
    '30%-<36%': (30.0, 36.0),
    '50%-60%': (50.0, 60.0),
    '<20%': (np.nan, 20.0),
    '>60%': (60.0, np.nan),
    'MISSING': (np.nan, np.nan),
}


def add_dti_bounds(frame):
    unknown = set(frame['debt_to_income_ratio'].unique()) - set(DTI_BOUNDS)
    require(not unknown, f'DTI 出现未知类别：{sorted(map(str, unknown))}')
    result = frame.copy()
    pairs = frame['debt_to_income_ratio'].map(DTI_BOUNDS)
    result['dti_lower'] = pairs.map(lambda pair: pair[0]).astype('float64')
    result['dti_upper'] = pairs.map(lambda pair: pair[1]).astype('float64')
    require(result.columns.tolist() == FEATURE_COLUMNS + ['dti_lower', 'dti_upper'],
            'DTI 特征列或顺序错误')
    require(result['debt_to_income_ratio'].equals(frame['debt_to_income_ratio']),
            '原始 DTI 类别被修改')
    return result


X_dti = add_dti_bounds(X_base)
X_test_dti = add_dti_bounds(X_test_base)
print(f'候选特征：原始 56 列 + 2 列 DTI 上下界 = {X_dti.shape[1]} 列。')
display(X_dti[['debt_to_income_ratio', 'dti_lower', 'dti_upper']].head())