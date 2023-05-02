import os
import glob
import pandas as pd
# 
# # 합치려는 CSV 파일들이 있는 디렉터리를 지정합니다.
# input_dir = "/home/siwon/dev/Deeplearning-6/data/data_carla"
# 
# # 모든 CSV 파일의 경로를 가져옵니다.
# csv_files = glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True)
# 
# # 데이터프레임들을 저장할 리스트를 만듭니다.
# dataframes = []
# 
# # 각 CSV 파일을 읽어들인 뒤, id 열을 겹치지 않게 조정합니다.
# max_id = 0
# for file in csv_files:
#     df = pd.read_csv(file)
#     df['id'] = df['id'] + max_id  # 겹치지 않도록 이전 최대 id를 더합니다.
#     max_id = df['id'].max()  # 현재 파일의 최대 id를 업데이트합니다.
#     dataframes.append(df)
# 
# # 데이터프레임들을 하나로 합칩니다.
# merged_data = pd.concat(dataframes, ignore_index=True)
# 
# # 합쳐진 데이터를 새로운 CSV 파일로 저장합니다.
# output_file = os.path.join(input_dir, "merged_data.csv")
# merged_data.to_csv(output_file, index=False)

def resize_points(df, max_points=470):
    # 각 id에 대해 반복하며
    unique_ids = df['id'].unique()
    resized_data = []

    for uid in unique_ids:
        # 현재 id에 해당하는 데이터만 추출
        current_data = df[df['id'] == uid]
        
        # 데이터가 max_points보다 작으면 패딩, 크면 자르기
        if len(current_data) < max_points:
            padding_size = max_points - len(current_data)
            padding_df = pd.DataFrame(index=range(padding_size), columns=current_data.columns)
            padding_df['id'] = uid
            padding_df['point_id'] = range(len(current_data), max_points)
            padding_df[['X', 'Y', 'Z']] = current_data.iloc[-1][['X', 'Y', 'Z']].values  # 마지막 위치로 채우기
            
            resized_df = pd.concat([current_data, padding_df], ignore_index=True)
        else:
            resized_df = current_data.iloc[:max_points]

        resized_data.append(resized_df)

    # 조정된 데이터를 합쳐서 반환
    resized_df = pd.concat(resized_data, ignore_index=True)
    return resized_df

# 데이터를 불러옵니다.
data = pd.read_csv("/home/siwon/dev/Deeplearning-6/data/data_carla/merged_data.csv")

# 각 id에 대해 point_id가 470개가 되도록 조정합니다.
resized_data = resize_points(data, max_points=470)

# 조정된 데이터를 새로운 CSV 파일로 저장합니다.
resized_data.to_csv("/home/siwon/dev/Deeplearning-6/data/data_carla/carla_data.csv", index=False)
