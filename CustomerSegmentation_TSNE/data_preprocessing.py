import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataPreprocessor:
    def __init__(self, data_path):
        """
        Khởi tạo bộ tiền xử lý dữ liệu
        
        Parameters:
        -----------
        data_path : str
            Đường dẫn đến file dữ liệu CSV
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.numerical_features = None
        self.categorical_features = None
        self.target = None
        self.scaler = None
        self.encoder = None
        
    def load_data(self):
        """
        Đọc dữ liệu từ file CSV
        
        Returns:
        --------
        pandas.DataFrame
            Dữ liệu thô
        """
        print(f"Đọc dữ liệu từ {self.data_path}")
        self.raw_data = pd.read_csv(self.data_path)
        return self.raw_data
    
    def explore_data(self):
        """
        Khám phá dữ liệu và trả về thông tin cơ bản
        
        Returns:
        --------
        dict
            Thông tin cơ bản về dữ liệu
        """
        if self.raw_data is None:
            self.load_data()
            
        # Tạo thư mục để lưu các biểu đồ
        os.makedirs('static/images/eda', exist_ok=True)
        
        # Thông tin cơ bản
        info = {
            'shape': self.raw_data.shape,
            'columns': self.raw_data.columns.tolist(),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'data_types': self.raw_data.dtypes.astype(str).to_dict(),
            'summary_stats': self.raw_data.describe().to_dict()
        }
        
        # Lưu thông tin phân phối của các biến số
        numeric_cols = self.raw_data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.raw_data[col], kde=True)
            plt.title(f'Phân phối của {col}')
            plt.savefig(f'static/images/eda/{col}_distribution.png')
            plt.close()
            
        # Lưu biểu đồ tần suất cho các biến phân loại
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            plt.figure(figsize=(12, 6))
            sns.countplot(y=col, data=self.raw_data, order=self.raw_data[col].value_counts().index)
            plt.title(f'Tần suất của {col}')
            plt.savefig(f'static/images/eda/{col}_frequency.png')
            plt.close()
            
        # Ma trận tương quan
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.raw_data.select_dtypes(include=['int64', 'float64']).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Ma trận tương quan')
        plt.savefig('static/images/eda/correlation_matrix.png')
        plt.close()
        
        return info
    
    def identify_features(self):
        """
        Xác định các đặc trưng số và đặc trưng phân loại
        """
        if self.raw_data is None:
            self.load_data()
            
        # Xác định các loại đặc trưng
        self.numerical_features = self.raw_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.raw_data.select_dtypes(include=['object']).columns.tolist()
        
        # Loại bỏ CustomerID khỏi các đặc trưng (nếu có)
        if 'CustomerID' in self.numerical_features:
            self.numerical_features.remove('CustomerID')
            
        print(f"Đặc trưng số: {self.numerical_features}")
        print(f"Đặc trưng phân loại: {self.categorical_features}")
        
    def handle_missing_values(self):
        """
        Xử lý giá trị thiếu trong dữ liệu
        
        Returns:
        --------
        pandas.DataFrame
            Dữ liệu đã xử lý giá trị thiếu
        """
        if self.raw_data is None:
            self.load_data()
            
        if self.numerical_features is None or self.categorical_features is None:
            self.identify_features()
            
        # Kiểm tra giá trị thiếu
        missing_values = self.raw_data.isnull().sum()
        print("Số lượng giá trị thiếu trong mỗi cột:")
        print(missing_values[missing_values > 0])
        
        # Xử lý giá trị thiếu cho đặc trưng số bằng KNN Imputer
        if any(missing_values[self.numerical_features] > 0):
            print("Áp dụng KNN Imputer cho đặc trưng số")
            imputer = KNNImputer(n_neighbors=5)
            self.raw_data[self.numerical_features] = imputer.fit_transform(self.raw_data[self.numerical_features])
        
        # Xử lý giá trị thiếu cho đặc trưng phân loại bằng giá trị phổ biến nhất
        for col in self.categorical_features:
            if missing_values[col] > 0:
                print(f"Điền giá trị phổ biến nhất cho {col}")
                most_frequent = self.raw_data[col].mode()[0]
                self.raw_data[col].fillna(most_frequent, inplace=True)
                
        return self.raw_data
    
    def detect_outliers(self, contamination=0.05):
        """
        Phát hiện và xử lý ngoại lai bằng Isolation Forest
        
        Parameters:
        -----------
        contamination : float, default=0.05
            Tỷ lệ ngoại lai dự kiến trong dữ liệu
            
        Returns:
        --------
        pandas.DataFrame
            Dữ liệu đã loại bỏ ngoại lai
        """
        if self.raw_data is None:
            self.load_data()
            
        if self.numerical_features is None:
            self.identify_features()
            
        # Chỉ áp dụng Isolation Forest cho các đặc trưng số
        print("Phát hiện ngoại lai bằng Isolation Forest")
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(self.raw_data[self.numerical_features])
        
        # Tạo cột đánh dấu ngoại lai (1: bình thường, -1: ngoại lai)
        self.raw_data['outlier'] = outlier_labels
        
        # Vẽ biểu đồ phân phối ngoại lai
        plt.figure(figsize=(10, 6))
        sns.countplot(x='outlier', data=self.raw_data)
        plt.title('Phân phối ngoại lai')
        plt.savefig('static/images/eda/outliers_distribution.png')
        plt.close()
        
        # Lưu dữ liệu không có ngoại lai
        clean_data = self.raw_data[self.raw_data['outlier'] == 1].drop('outlier', axis=1)
        outliers = self.raw_data[self.raw_data['outlier'] == -1].drop('outlier', axis=1)
        
        print(f"Số lượng mẫu ban đầu: {len(self.raw_data)}")
        print(f"Số lượng mẫu sau khi loại bỏ ngoại lai: {len(clean_data)}")
        print(f"Số lượng ngoại lai đã phát hiện: {len(outliers)}")
        
        # Cập nhật dữ liệu
        self.raw_data = clean_data
        
        return clean_data, outliers
    
    def encode_categorical_features(self):
        """
        Mã hóa các đặc trưng phân loại
        
        Returns:
        --------
        pandas.DataFrame
            Dữ liệu đã mã hóa
        """
        if self.raw_data is None:
            self.load_data()
            
        if self.categorical_features is None:
            self.identify_features()
            
        # Tạo bản sao của dữ liệu
        encoded_data = self.raw_data.copy()
        
        # Mã hóa các đặc trưng phân loại bằng OneHotEncoder
        print("Mã hóa các đặc trưng phân loại")
        for col in self.categorical_features:
            # Sử dụng pd.get_dummies để one-hot encoding
            dummies = pd.get_dummies(encoded_data[col], prefix=col, drop_first=True)
            encoded_data = pd.concat([encoded_data, dummies], axis=1)
            encoded_data.drop(col, axis=1, inplace=True)
            
        self.processed_data = encoded_data
        return encoded_data
    
    def scale_numerical_features(self):
        """
        Chuẩn hóa các đặc trưng số
        
        Returns:
        --------
        pandas.DataFrame
            Dữ liệu đã chuẩn hóa
        """
        if self.processed_data is None:
            self.encode_categorical_features()
            
        # Tạo bản sao của dữ liệu
        scaled_data = self.processed_data.copy()
        
        # Chuẩn hóa các đặc trưng số bằng StandardScaler
        print("Chuẩn hóa các đặc trưng số")
        scaler = StandardScaler()
        scaled_data[self.numerical_features] = scaler.fit_transform(scaled_data[self.numerical_features])
        
        self.scaler = scaler
        self.processed_data = scaled_data
        return scaled_data
    
    def create_feature_pipeline(self):
        """
        Tạo pipeline xử lý đặc trưng
        
        Returns:
        --------
        sklearn.pipeline.Pipeline
            Pipeline xử lý đặc trưng
        """
        if self.numerical_features is None or self.categorical_features is None:
            self.identify_features()
            
        # Tạo pipeline cho đặc trưng số
        numerical_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        # Tạo pipeline cho đặc trưng phân loại
        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Kết hợp các pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor
    
    def preprocess_data(self):
        """
        Tiền xử lý dữ liệu hoàn chỉnh
        
        Returns:
        --------
        pandas.DataFrame
            Dữ liệu đã tiền xử lý
        """
        print("Bắt đầu tiền xử lý dữ liệu...")
        
        # Đọc dữ liệu
        self.load_data()
        
        # Khám phá dữ liệu
        self.explore_data()
        
        # Xác định đặc trưng
        self.identify_features()
        
        # Xử lý giá trị thiếu
        self.handle_missing_values()
        
        # Phát hiện và xử lý ngoại lai
        self.detect_outliers()
        
        # Mã hóa đặc trưng phân loại
        self.encode_categorical_features()
        
        # Chuẩn hóa đặc trưng số
        self.scale_numerical_features()
        
        print("Hoàn thành tiền xử lý dữ liệu!")
        return self.processed_data
    
    def save_processed_data(self, output_path):
        """
        Lưu dữ liệu đã tiền xử lý
        
        Parameters:
        -----------
        output_path : str
            Đường dẫn để lưu dữ liệu đã tiền xử lý
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        self.processed_data.to_csv(output_path, index=False)
        print(f"Đã lưu dữ liệu đã tiền xử lý tại {output_path}")


if __name__ == "__main__":
    # Ví dụ sử dụng
    data_path = "../customer_São_Paulo_2024.csv"
    preprocessor = DataPreprocessor(data_path)
    processed_data = preprocessor.preprocess_data()
    preprocessor.save_processed_data("../processed_customer_data.csv")
