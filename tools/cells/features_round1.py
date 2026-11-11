def safe_ratio(numerator, denominator):
    valid = numerator.notna() & denominator.notna() & (denominator > 0)
    return (numerator / denominator.where(valid)).where(valid).astype('float64')


def build_blocks(train, test):
    both = pd.concat([train, test], ignore_index=True)
    blocks = {}
    blocks['dti'] = add_dti_bounds(both)[['dti_lower', 'dti_upper']]

    ltv_calc = 100 * safe_ratio(both['loan_amount'], both['property_value'])
    blocks['ltv'] = pd.DataFrame({
        'ltv_calc': ltv_calc,
        'ltv_gap': both['loan_to_value_ratio'] - ltv_calc,
    })

    income_dollars = both['income'] * 1000      # Les revenus HMDA sont exprimés en milliers de dollars
    blocks['income'] = pd.DataFrame({
        'income_to_area': safe_ratio(income_dollars, both['ffiec_msa_md_median_family_income']),
        'loan_to_income': safe_ratio(both['loan_amount'], income_dollars),
        'value_to_income': safe_ratio(both['property_value'], income_dollars),
    })

    missing = pd.DataFrame({
        'ltv': both['loan_to_value_ratio'].isna(),
        'pv': both['property_value'].isna(),
        'inc': both['income'].isna(),
        'term': both['loan_term'].isna(),
        'dti': both['debt_to_income_ratio'].eq('MISSING'),
        'age': both['applicant_age'].eq('MISSING'),
    })
    blocks['missing'] = pd.DataFrame({
        'missing_pattern': 'm' + missing.astype(int).astype(str).agg(''.join, axis=1),
        'missing_count': missing.sum(axis=1).astype('float64'),
    })

    freq = {}
    for column in ['county_code', 'derived_msa-md', 'state_code']:
        counts = both[column].map(both[column].value_counts()).astype('float64')
        freq[f'freq_{column}'] = counts.where(both[column].ne('MISSING'))
    blocks['freq'] = pd.DataFrame(freq)

    n_train = len(train)
    out = {}
    for name, frame in blocks.items():
        require(len(frame) == len(both), f'{name} : nombre de lignes incorrect')
        out[name] = (frame.iloc[:n_train].reset_index(drop=True),
                     frame.iloc[n_train:].reset_index(drop=True))
    return out


BLOCKS = build_blocks(X_base, X_test_base)
NEW_CATEGORICAL = {'missing_pattern'}


def variant_frames(blocks):
    train_parts = [X_base] + [BLOCKS[b][0] for b in blocks]
    test_parts = [X_test_base] + [BLOCKS[b][1] for b in blocks]
    train = pd.concat(train_parts, axis=1)
    test = pd.concat(test_parts, axis=1)
    require(train.columns.is_unique and train.columns.equals(test.columns), 'Colonnes incohérentes entre variantes')
    require(train.iloc[:, :len(FEATURE_COLUMNS)].equals(X_base), 'Les 56 colonnes initiales ont été modifiées')
    cats = CATEGORICAL_COLUMNS + [c for c in train.columns if c in NEW_CATEGORICAL]
    return train, test, cats


for name, (train_block, _) in BLOCKS.items():
    described = train_block.describe(include='all').T
    print(f'\n[{name}] Taux de valeurs manquantes : ', train_block.isna().mean().round(3).to_dict())
    display_named(described.iloc[:, :6])
