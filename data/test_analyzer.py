from file_analyzer import process_uploaded_file

df, df_raw, mapping, table_name, quality_report = process_uploaded_file('test_orders.csv')

print('===== Column Mapping =====')
for original, mapped in mapping.items():
    print(f'{original} --> {mapped}')

print('\n===== Data Quality Report =====')
for k, v in quality_report.items():
    print(f'{k}: {v}')

print('\n===== Cleaned Data =====')
print(df)
print(f'\nRows after cleaning: {len(df)}')