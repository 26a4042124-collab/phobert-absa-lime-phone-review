#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script để đọc và hiển thị dữ liệu CSV theo cột dễ đọc
Cách dùng: python view_data.py [tên_file.csv] [số_dòng]
Ví dụ: python view_data.py train.csv 10
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Đường dẫn tới thư mục data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

def main():
    if len(sys.argv) < 2:
        print("Cách sử dụng: python view_data.py <tên_file.csv> [số_dòng]")
        print("\nCác file có sẵn:")
        if os.path.exists(DATA_DIR):
            for file in os.listdir(DATA_DIR):
                if file.endswith('.csv'):
                    print(f"  - {file}")
        print("\nVí dụ:")
        print("  python view_data.py train.csv")
        print("  python view_data.py train.csv 5")
        return
    
    filename = sys.argv[1]
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return
    
    try:
        # Đọc file CSV
        df = pd.read_csv(file_path)
        
        print(f"\n{'='*100}")
        print(f"📄 File: {filename}")
        print(f"{'='*100}")
        print(f"\n📊 Kích thước: {df.shape[0]} dòng × {df.shape[1]} cột")
        
        # Hiển thị tên các cột
        print(f"\n📋 Các cột:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col:<20} ({str(df[col].dtype):<15})")
        
        # Hiển thị dữ liệu
        print(f"\n📑 Dữ liệu:")
        print("-" * 100)
        
        # Set pandas options để hiển thị đầy đủ
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 100)
        
        if rows:
            df_display = df.head(rows)
            print(f"Hiển thị {rows} dòng đầu tiên:")
        else:
            df_display = df
            print(f"Hiển thị tất cả {len(df)} dòng:")
        
        print()
        print(df_display.to_string())
        
        # Hiển thị thống kê
        print(f"\n\n📊 Thống kê mô tả:")
        print("-" * 100)
        print(df.describe(include='all').to_string())
        
        print(f"\n{'='*100}\n")
        
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")

if __name__ == "__main__":
    main()
