import pandas as pd
import time
import numpy as np
import os
import argparse
from typing import Dict, List, Tuple
from glob import glob
from tqdm import tqdm

COLS = [
    "운행일자",
    "일련번호",
    "가상카드번호",
    "정산지역코드",
    "카드구분코드",
    "차량ID",
    "차량등록번호",
    "운행출발일시",
    "운행종료일시",
    "교통수단코드",
    "노선ID(국토부표준)",
    "노선ID(정산사업자)",
    "승차일시",
    "발권시간",
    "승차정류장ID(국토부표준)",
    "승차정류장ID(정산사업자)",
    "하차일시",
    "하차정류장ID(국토부표준)",
    "하차정류장ID(정산사업자)",
    "트랜잭션ID",
    "환승횟수",
    "사용자구분코드",
    "이용객수",
    "이용거리",
    "탑승시간",
]

def numberic_norm(s):
    s = s.astype("string")  # pandas StringDtype
    s = s.str.strip()
    return s.str.zfill(4)


def mapping_log(log, metro_df):
    map_name = dict(zip(metro_df["역번호"],metro_df["노드역명"]))
    log["승차역명"] = log["승차정류장ID(정산사업자)"].map(map_name)
    log["하차역명"] = log["하차정류장ID(정산사업자)"].map(map_name) 
    return log


def date_log(log):
    log["승차일시"] = pd.to_datetime(log["승차일시"], format="%Y%m%d%H%M%S")
    log["하차일시"] = pd.to_datetime(log["하차일시"], format="%Y%m%d%H%M%S")
    log["이동시간"] = log["하차일시"] - log["승차일시"]
    log["이동시간"] = log["이동시간"].apply(
        lambda x: f"{x.seconds//3600:02}:{(x.seconds%3600)//60:02}"
    )

    # 읽기편하게 str 타입으로 해놨는데 필요없으면 제거 해도 됨
    log["승차일시"] = log["승차일시"].dt.strftime("%Y-%m-%d %H:%M")
    log["하차일시"] = log["하차일시"].dt.strftime("%Y-%m-%d %H:%M")  
    # log["이동시간"] = log["이동시간"].dt.strftime("%H:%M")

def processing_log(src_path, out_path, metro_df):
    os.makedirs(out_path,exist_ok=True)
    txt_files = [f for f in os.listdir(src_path)]
    metro_num = set(str(num) for num in range(201,238)) ## 지하철 교통수단 코드 201~~237
    line_2 = set(str(num).zfill(4) for num in range(201,251)) ## 2호선 역번호 코드 201~250

    # 전체 파일 수와 총 용량 계산
    total_files = len(txt_files)
    total_size = 0
    for txt in txt_files:
        file_path = os.path.join(src_path, txt)
        if os.path.exists(file_path):
            total_size += os.path.getsize(file_path)
    
    # 용량 MB 단위로 변환
    total_size_mb = total_size / (1024 * 1024)

    print(f"Total files: {total_files}개")
    print(f"Total size: {total_size_mb:.2f} MB")
    print("=" * 50)

    # 프로그레스바와 함께 파일 처리
    processed_count = 0

    
    for txt in tqdm(txt_files, desc="File Processing..", unit="file"):
        try: 
            # 현재 파일 정보
            file_path = os.path.join(src_path, txt)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            df = pd.read_csv(
                file_path,
                sep="|",
                header=None,
                dtype=str,
                na_values=["", " ", "NA", "NaN", "nan"],
                names = COLS
            )
            
            mask = np.isin(df.iloc[:,9 ],list(metro_num)) # 교통수단코드 9번 열
            origin = np.isin(df.iloc[:,15],list(line_2)) # 승차정류장ID(정산사업자) 15번 열
            dest = np.isin(df.iloc[:,18],list(line_2)) # 하차정류장ID(정산사업자) 18번 열

            df = df[mask & (origin | dest )]   
            
            df["승차정류장ID(정산사업자)"] = numberic_norm(df["승차정류장ID(정산사업자)"])
            df["하차정류장ID(정산사업자)"] = numberic_norm(df["하차정류장ID(정산사업자)"])
            # 필요한 열만 선택하여 df를 필터링
            filter = [
                "가상카드번호",
                "승차정류장ID(정산사업자)", "하차정류장ID(정산사업자)",
                "승차일시", "하차일시","트랜잭션ID","환승횟수"
            ]
            df = df[filter]
            mapped_log = mapping_log(df, metro_df)
            date_log(mapped_log)
            
            # 저장할 파일명을 지정
            save_name = f"{os.path.splitext(txt)[0]}.parquet"
            save_path = os.path.join(out_path, save_name)
            mapped_log.to_parquet(save_path, engine="pyarrow", compression='gzip')
            
            processed_count += 1
        
            tqdm.write(f"[{processed_count}/{total_files}] {txt} Process completed - "
                       f"(File size: {file_size_mb:.2f}MB)")
                       
        except Exception as e:
            tqdm.write(f"❌ {txt} Process failed: {str(e)}")
            continue
    
    print("=" * 50)
    print(f"✅ All files processing finished! ({processed_count}/{total_files} files processed)")

def main():
    parser = argparse.ArgumentParser(description="지하철 데이터 처리 스크립트")
    parser.add_argument("--src", required=True, help="input TXT folder path")
    parser.add_argument("--out", required=True, help="output Parquet folder path")
    parser.add_argument("--line_info", help="line info xlsx file path")
    args = parser.parse_args()

    try:        
        metro = np.load("./metro.npy",allow_pickle=True)
    except:
        if args.line_info is None:
            raise ValueError("line_info xlsx file path is required when metro.npy does not exist.")
        df = pd.read_excel(args.line_info, header=0)
        np.save("metro.npy",df[["교통수단코드","호선코드","호선명","역번호","노드역명"]])
        metro = np.load("./metro.npy",allow_pickle=True)
    
    metro = metro[metro[:,0].argsort()] #metro 데이터를 교통수단코드순으로 정렬
    metro_df = pd.DataFrame(
        metro,
        columns=["교통수단코드","호선코드","호선명","역번호","노드역명"]
    )
    processing_log(args.src, args.out, metro_df)

    df = pd.read_parquet(f"{args.out}/TCD_20220501.parquet").sort_values(["승차일시"])
    print(df.head(20))

if __name__ == "__main__":
    main()