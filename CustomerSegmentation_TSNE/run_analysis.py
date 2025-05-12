"""
Script chạy thử phân tích khách hàng với t-SNE và K-means
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor
from clustering import CustomerClustering

def main():
    """
    Hàm chính để chạy toàn bộ quy trình phân tích
    """
    print("="*50)
    print("PHÂN TÍCH KHÁCH HÀNG VỚI t-SNE VÀ K-MEANS")
    print("="*50)
    
    # Đường dẫn đến dữ liệu
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'customer_São_Paulo_2024.csv')
    processed_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed_customer_data.csv')
    
    # Kiểm tra xem dữ liệu đã được tiền xử lý chưa
    if os.path.exists(processed_data_path):
        print(f"Dữ liệu đã tiền xử lý được tìm thấy tại: {processed_data_path}")
        preprocess_again = input("Bạn có muốn tiền xử lý lại dữ liệu không? (y/n): ")
        
        if preprocess_again.lower() == 'y':
            run_preprocessing(data_path, processed_data_path)
    else:
        print(f"Không tìm thấy dữ liệu đã tiền xử lý. Tiến hành tiền xử lý...")
        run_preprocessing(data_path, processed_data_path)
    
    # Chạy phân cụm
    run_clustering(processed_data_path)
    
    print("\n"+"="*50)
    print("PHÂN TÍCH HOÀN THÀNH!")
    print("="*50)
    print("\nBạn có thể xem kết quả chi tiết bằng cách chạy ứng dụng web:")
    print("python app.py")

def run_preprocessing(data_path, output_path):
    """
    Tiền xử lý dữ liệu
    
    Parameters:
    -----------
    data_path : str
        Đường dẫn đến file dữ liệu gốc
    output_path : str
        Đường dẫn để lưu dữ liệu đã tiền xử lý
    """
    print("\n"+"="*50)
    print("TIỀN XỬ LÝ DỮ LIỆU")
    print("="*50)
    
    # Khởi tạo bộ tiền xử lý dữ liệu
    preprocessor = DataPreprocessor(data_path)
    
    # Tiền xử lý dữ liệu
    processed_data = preprocessor.preprocess_data()
    
    # Lưu dữ liệu đã tiền xử lý
    preprocessor.save_processed_data(output_path)
    
    print(f"\nTiền xử lý dữ liệu hoàn thành! Dữ liệu đã được lưu tại: {output_path}")
    
    return processed_data

def run_clustering(data_path, perplexity=30, n_clusters=None):
    """
    Phân cụm khách hàng
    
    Parameters:
    -----------
    data_path : str
        Đường dẫn đến file dữ liệu đã tiền xử lý
    perplexity : float, default=30
        Perplexity cho t-SNE
    n_clusters : int, default=None
        Số cụm cho K-means. Nếu None, sẽ tự động tìm số cụm tối ưu
    """
    print("\n"+"="*50)
    print("PHÂN CỤM KHÁCH HÀNG VỚI t-SNE VÀ K-MEANS")
    print("="*50)
    
    # Khởi tạo đối tượng phân cụm
    clustering = CustomerClustering(data_path=data_path)
    
    # Chạy phân tích hoàn chỉnh
    print(f"\nBắt đầu phân tích với perplexity={perplexity}")
    if n_clusters is None:
        print("Tự động tìm số cụm tối ưu...")
    else:
        print(f"Số cụm: {n_clusters}")
        
    results = clustering.run_complete_analysis(
        perplexity=perplexity,
        n_clusters=n_clusters
    )
    
    # In kết quả
    print("\nKết quả phân cụm:")
    print(f"Số cụm tối ưu: {results['optimal_clusters']}")
    
    # In so sánh t-SNE và PCA
    print("\nSo sánh t-SNE và PCA:")
    print(results['comparison'])
    
    # In đặc điểm của các cụm
    print("\nĐặc điểm trung bình của các phân khúc khách hàng:")
    print(results['cluster_features'])
    
    print("\nPhân cụm hoàn thành!")
    print(f"Các biểu đồ đã được lưu trong thư mục: static/images/clustering/")
    
    return results

if __name__ == "__main__":
    main()
