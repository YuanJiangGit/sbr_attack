import pandas as pd

path = 'C:\\Users\\shan\\Desktop\\all.csv'
df = pd.read_csv(path)
full_score_df = df[df['score'] == df['q_score']]
result_df = full_score_df.groupby('q_id').size().sort_values(ascending=False)
output_file_path = 'C:\\Users\\shan\\Desktop\\result.csv'
# for p_id, size in result_df.items():
result_df.to_csv(output_file_path, encoding='utf-8')
