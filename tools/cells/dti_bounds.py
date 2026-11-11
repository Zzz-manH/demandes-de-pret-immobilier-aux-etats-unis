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
    require(not unknown, f'Catégories DTI inconnues : {sorted(map(str, unknown))}')
    result = frame.copy()
    pairs = frame['debt_to_income_ratio'].map(DTI_BOUNDS)
    result['dti_lower'] = pairs.map(lambda pair: pair[0]).astype('float64')
    result['dti_upper'] = pairs.map(lambda pair: pair[1]).astype('float64')
    require(result.columns.tolist() == FEATURE_COLUMNS + ['dti_lower', 'dti_upper'],
            'Colonnes DTI ou ordre incorrects')
    require(result['debt_to_income_ratio'].equals(frame['debt_to_income_ratio']),
            'Les catégories DTI initiales ont été modifiées')
    return result


X_dti = add_dti_bounds(X_base)
X_test_dti = add_dti_bounds(X_test_base)
print(f'Variables candidates : 56 colonnes initiales + 2 bornes DTI = {X_dti.shape[1]} colonnes.')
display_named(X_dti[['debt_to_income_ratio', 'dti_lower', 'dti_upper']].head())