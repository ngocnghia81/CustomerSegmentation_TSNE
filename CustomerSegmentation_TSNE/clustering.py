import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
from sklearn.preprocessing import StandardScaler

class CustomerClustering:
    def __init__(self, data=None, data_path=None):
        """
        Khởi tạo đối tượng phân cụm khách hàng
        
        Parameters:
        -----------
        data : pandas.DataFrame, default=None
            Dữ liệu đã tiền xử lý
        data_path : str, default=None
            Đường dẫn đến file dữ liệu đã tiền xử lý
        """
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = pd.read_csv(data_path)
        else:
            raise ValueError("Phải cung cấp dữ liệu hoặc đường dẫn đến file dữ liệu")
            
        self.features = None
        self.tsne_results = None
        self.pca_results = None
        self.kmeans_model = None
        self.optimal_clusters = None
        self.cluster_labels = None
        self.customer_segments = None
        
        # Tạo thư mục để lưu các biểu đồ và mô hình
        os.makedirs('static/images/clustering', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
    def prepare_features(self, exclude_cols=None):
        """
        Chuẩn bị các đặc trưng cho phân cụm
        
        Parameters:
        -----------
        exclude_cols : list, default=None
            Danh sách các cột cần loại trừ
            
        Returns:
        --------
        numpy.ndarray
            Ma trận đặc trưng
        """
        if exclude_cols is None:
            exclude_cols = ['CustomerID']
            
        # Loại bỏ các cột không cần thiết
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        self.features = self.data[feature_cols].values
        
        return self.features
    
    def apply_tsne(self, n_components=2, perplexity=30, learning_rate=200, max_iter=1000):
        """
        Áp dụng t-SNE để giảm chiều dữ liệu
        
        Parameters:
        -----------
        n_components : int, default=2
            Số chiều đầu ra
        perplexity : float, default=30
            Perplexity cho t-SNE
        learning_rate : float, default=200
            Tốc độ học
        max_iter : int, default=1000
            Số vòng lặp tối đa
            
        Returns:
        --------
        numpy.ndarray
            Kết quả t-SNE
        """
        if self.features is None:
            self.prepare_features()
            
        print(f"Áp dụng t-SNE với perplexity={perplexity}, learning_rate={learning_rate}")
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=42
        )
        
        self.tsne_results = tsne.fit_transform(self.features)
        
        # Lưu kết quả t-SNE vào DataFrame
        tsne_df = pd.DataFrame(
            self.tsne_results,
            columns=[f'TSNE{i+1}' for i in range(n_components)]
        )
        
        return self.tsne_results, tsne_df
    
    def apply_pca(self, n_components=2):
        """
        Áp dụng PCA để giảm chiều dữ liệu
        
        Parameters:
        -----------
        n_components : int, default=2
            Số thành phần chính
            
        Returns:
        --------
        numpy.ndarray
            Kết quả PCA
        """
        if self.features is None:
            self.prepare_features()
            
        print(f"Áp dụng PCA với n_components={n_components}")
        pca = PCA(n_components=n_components, random_state=42)
        self.pca_results = pca.fit_transform(self.features)
        
        # Lưu kết quả PCA vào DataFrame
        pca_df = pd.DataFrame(
            self.pca_results,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Tính tỷ lệ phương sai giải thích được
        explained_variance = pca.explained_variance_ratio_
        print(f"Tỷ lệ phương sai giải thích được: {explained_variance}")
        print(f"Tổng phương sai giải thích được: {sum(explained_variance):.4f}")
        
        # Vẽ biểu đồ scree plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), explained_variance)
        plt.plot(range(1, n_components + 1), np.cumsum(explained_variance), 'r-o')
        plt.xlabel('Thành phần chính')
        plt.ylabel('Tỷ lệ phương sai giải thích được')
        plt.title('Scree Plot')
        plt.savefig('static/images/clustering/pca_scree_plot.png')
        plt.close()
        
        return self.pca_results, pca_df, explained_variance
    
    def find_optimal_clusters(self, max_clusters=10, method='silhouette'):
        """
        Tìm số cụm tối ưu
        
        Parameters:
        -----------
        max_clusters : int, default=10
            Số cụm tối đa cần xem xét
        method : str, default='silhouette'
            Phương pháp để xác định số cụm tối ưu
            
        Returns:
        --------
        int
            Số cụm tối ưu
        """
        if self.tsne_results is None:
            self.apply_tsne()
            
        print(f"Tìm số cụm tối ưu với phương pháp {method}")
        
        # Sử dụng kết quả t-SNE để tìm số cụm tối ưu
        X = self.tsne_results
        
        if method == 'elbow':
            # Phương pháp Elbow
            plt.figure(figsize=(10, 6))
            visualizer = KElbowVisualizer(KMeans(random_state=42), k=(2, max_clusters))
            visualizer.fit(X)
            visualizer.finalize()
            plt.savefig('static/images/clustering/elbow_method.png')
            plt.close()
            
            self.optimal_clusters = visualizer.elbow_value_
            
        elif method == 'silhouette':
            # Phương pháp Silhouette Score
            silhouette_scores = []
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                score = silhouette_score(X, cluster_labels)
                silhouette_scores.append(score)
                print(f"Silhouette Score với {k} cụm: {score:.4f}")
                
            # Vẽ biểu đồ silhouette score
            plt.figure(figsize=(10, 6))
            plt.plot(range(2, max_clusters + 1), silhouette_scores, 'bo-')
            plt.xlabel('Số cụm')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Method')
            plt.grid(True)
            plt.savefig('static/images/clustering/silhouette_method.png')
            plt.close()
            
            # Số cụm tối ưu là số cụm có silhouette score cao nhất
            self.optimal_clusters = np.argmax(silhouette_scores) + 2
            
        else:
            raise ValueError("Phương pháp không hợp lệ. Chọn 'elbow' hoặc 'silhouette'")
            
        print(f"Số cụm tối ưu: {self.optimal_clusters}")
        return self.optimal_clusters
    
    def apply_kmeans(self, n_clusters=None):
        """
        Áp dụng thuật toán K-means
        
        Parameters:
        -----------
        n_clusters : int, default=None
            Số cụm. Nếu None, sẽ sử dụng số cụm tối ưu đã tìm được
            
        Returns:
        --------
        numpy.ndarray
            Nhãn cụm
        """
        if self.tsne_results is None:
            self.apply_tsne()
            
        if n_clusters is None:
            if self.optimal_clusters is None:
                self.find_optimal_clusters()
            n_clusters = self.optimal_clusters
            
        print(f"Áp dụng K-means với {n_clusters} cụm")
        
        # Áp dụng K-means trên kết quả t-SNE
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.tsne_results)
        self.kmeans_model = kmeans
        
        # Lưu mô hình
        joblib.dump(kmeans, 'models/kmeans_model.pkl')
        
        # Vẽ biểu đồ silhouette
        plt.figure(figsize=(10, 8))
        visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
        visualizer.fit(self.tsne_results)
        visualizer.finalize()
        plt.savefig('static/images/clustering/silhouette_visualization.png')
        plt.close()
        
        return self.cluster_labels
    
    def visualize_clusters_2d(self):
        """
        Trực quan hóa các cụm trong không gian 2D
        
        Returns:
        --------
        plotly.graph_objects.Figure
            Biểu đồ trực quan hóa
        """
        if self.cluster_labels is None:
            self.apply_kmeans()
            
        if self.tsne_results is None:
            self.apply_tsne()
            
        if self.pca_results is None:
            self.apply_pca()
            
        # Tạo DataFrame cho trực quan hóa
        tsne_df = pd.DataFrame(
            self.tsne_results,
            columns=['TSNE1', 'TSNE2']
        )
        tsne_df['Cluster'] = self.cluster_labels
        
        pca_df = pd.DataFrame(
            self.pca_results,
            columns=['PC1', 'PC2']
        )
        pca_df['Cluster'] = self.cluster_labels
        
        # Tạo biểu đồ t-SNE với Plotly
        fig_tsne = px.scatter(
            tsne_df, x='TSNE1', y='TSNE2',
            color='Cluster',
            color_continuous_scale=px.colors.qualitative.G10,
            title='Phân cụm khách hàng với t-SNE và K-means',
            labels={'Cluster': 'Phân khúc'},
            hover_data=['TSNE1', 'TSNE2', 'Cluster']
        )
        
        # Tạo biểu đồ PCA với Plotly
        fig_pca = px.scatter(
            pca_df, x='PC1', y='PC2',
            color='Cluster',
            color_continuous_scale=px.colors.qualitative.G10,
            title='Phân cụm khách hàng với PCA và K-means',
            labels={'Cluster': 'Phân khúc'},
            hover_data=['PC1', 'PC2', 'Cluster']
        )
        
        # Lưu biểu đồ
        fig_tsne.write_html('static/images/clustering/tsne_clusters.html')
        fig_pca.write_html('static/images/clustering/pca_clusters.html')
        
        # Tạo biểu đồ so sánh
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('t-SNE + K-means', 'PCA + K-means')
        )
        
        # Thêm biểu đồ t-SNE
        for cluster in tsne_df['Cluster'].unique():
            cluster_data = tsne_df[tsne_df['Cluster'] == cluster]
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['TSNE1'],
                    y=cluster_data['TSNE2'],
                    mode='markers',
                    name=f'Cluster {cluster} (t-SNE)',
                    marker=dict(size=8),
                    showlegend=True
                ),
                row=1, col=1
            )
            
        # Thêm biểu đồ PCA
        for cluster in pca_df['Cluster'].unique():
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['PC1'],
                    y=cluster_data['PC2'],
                    mode='markers',
                    name=f'Cluster {cluster} (PCA)',
                    marker=dict(size=8),
                    showlegend=True
                ),
                row=1, col=2
            )
            
        # Cập nhật layout
        fig.update_layout(
            title_text='So sánh phân cụm với t-SNE và PCA',
            height=600,
            width=1200
        )
        
        # Lưu biểu đồ so sánh
        fig.write_html('static/images/clustering/comparison_clusters.html')
        
        return fig_tsne, fig_pca, fig
    
    def analyze_clusters(self):
        """
        Phân tích đặc điểm của các cụm
        
        Returns:
        --------
        pandas.DataFrame
            Thông tin về các cụm
        """
        if self.cluster_labels is None:
            self.apply_kmeans()
            
        # Thêm nhãn cụm vào dữ liệu gốc
        data_with_clusters = self.data.copy()
        data_with_clusters['Cluster'] = self.cluster_labels
        
        # Phân tích kích thước của các cụm
        cluster_sizes = data_with_clusters['Cluster'].value_counts().sort_index()
        print("Kích thước của các cụm:")
        print(cluster_sizes)
        
        # Vẽ biểu đồ kích thước cụm
        plt.figure(figsize=(10, 6))
        cluster_sizes.plot(kind='bar')
        plt.xlabel('Cụm')
        plt.ylabel('Số lượng khách hàng')
        plt.title('Kích thước của các cụm')
        plt.savefig('static/images/clustering/cluster_sizes.png')
        plt.close()
        
        # Phân tích đặc điểm của các cụm
        cluster_features = data_with_clusters.groupby('Cluster').mean()
        print("\nĐặc điểm trung bình của các cụm:")
        print(cluster_features)
        
        # Vẽ biểu đồ heatmap cho đặc điểm cụm
        plt.figure(figsize=(14, 10))
        sns.heatmap(cluster_features, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Đặc điểm trung bình của các cụm')
        plt.savefig('static/images/clustering/cluster_features_heatmap.png')
        plt.close()
        
        # Vẽ biểu đồ radar cho đặc điểm cụm
        # Chuẩn hóa dữ liệu để vẽ biểu đồ radar
        scaler = StandardScaler()
        cluster_features_scaled = pd.DataFrame(
            scaler.fit_transform(cluster_features),
            index=cluster_features.index,
            columns=cluster_features.columns
        )
        
        # Vẽ biểu đồ radar cho mỗi cụm
        for cluster in cluster_features_scaled.index:
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=cluster_features_scaled.loc[cluster].values,
                theta=cluster_features_scaled.columns,
                fill='toself',
                name=f'Cluster {cluster}'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[-2, 2]
                    )
                ),
                title=f'Đặc điểm của Cluster {cluster}'
            )
            
            fig.write_html(f'static/images/clustering/radar_cluster_{cluster}.html')
        
        # Lưu thông tin về các cụm
        self.customer_segments = data_with_clusters
        self.customer_segments.to_csv('static/customer_segments.csv', index=False)
        
        return cluster_features, data_with_clusters
    
    def compare_tsne_pca(self):
        """
        So sánh hiệu quả của t-SNE và PCA
        
        Returns:
        --------
        dict
            Kết quả so sánh
        """
        if self.tsne_results is None:
            self.apply_tsne()
            
        if self.pca_results is None:
            self.apply_pca()
            
        if self.cluster_labels is None:
            self.apply_kmeans()
            
        # Tính các chỉ số đánh giá cho t-SNE
        tsne_silhouette = silhouette_score(self.tsne_results, self.cluster_labels)
        tsne_db = davies_bouldin_score(self.tsne_results, self.cluster_labels)
        tsne_ch = calinski_harabasz_score(self.tsne_results, self.cluster_labels)
        
        # Tính các chỉ số đánh giá cho PCA
        pca_silhouette = silhouette_score(self.pca_results, self.cluster_labels)
        pca_db = davies_bouldin_score(self.pca_results, self.cluster_labels)
        pca_ch = calinski_harabasz_score(self.pca_results, self.cluster_labels)
        
        # Tạo bảng so sánh
        comparison = {
            'Phương pháp': ['t-SNE', 'PCA'],
            'Silhouette Score': [tsne_silhouette, pca_silhouette],
            'Davies-Bouldin Index': [tsne_db, pca_db],
            'Calinski-Harabasz Index': [tsne_ch, pca_ch]
        }
        
        comparison_df = pd.DataFrame(comparison)
        print("\nSo sánh t-SNE và PCA:")
        print(comparison_df)
        
        # Vẽ biểu đồ so sánh
        plt.figure(figsize=(12, 8))
        
        # Silhouette Score (cao hơn tốt hơn)
        plt.subplot(1, 3, 1)
        plt.bar(['t-SNE', 'PCA'], [tsne_silhouette, pca_silhouette])
        plt.title('Silhouette Score\n(cao hơn tốt hơn)')
        plt.ylim(0, 1)
        
        # Davies-Bouldin Index (thấp hơn tốt hơn)
        plt.subplot(1, 3, 2)
        plt.bar(['t-SNE', 'PCA'], [tsne_db, pca_db])
        plt.title('Davies-Bouldin Index\n(thấp hơn tốt hơn)')
        
        # Calinski-Harabasz Index (cao hơn tốt hơn)
        plt.subplot(1, 3, 3)
        plt.bar(['t-SNE', 'PCA'], [tsne_ch, pca_ch])
        plt.title('Calinski-Harabasz Index\n(cao hơn tốt hơn)')
        
        plt.tight_layout()
        plt.savefig('static/images/clustering/tsne_pca_comparison.png')
        plt.close()
        
        return comparison_df
    
    def run_complete_analysis(self, perplexity=30, n_clusters=None):
        """
        Chạy phân tích hoàn chỉnh
        
        Parameters:
        -----------
        perplexity : float, default=30
            Perplexity cho t-SNE
        n_clusters : int, default=None
            Số cụm cho K-means
            
        Returns:
        --------
        dict
            Kết quả phân tích
        """
        print("Bắt đầu phân tích hoàn chỉnh...")
        
        # Chuẩn bị đặc trưng
        self.prepare_features()
        
        # Áp dụng t-SNE
        self.apply_tsne(perplexity=perplexity)
        
        # Áp dụng PCA
        self.apply_pca()
        
        # Tìm số cụm tối ưu nếu không được chỉ định
        if n_clusters is None:
            self.find_optimal_clusters()
        else:
            self.optimal_clusters = n_clusters
            
        # Áp dụng K-means
        self.apply_kmeans(n_clusters=self.optimal_clusters)
        
        # Trực quan hóa các cụm
        self.visualize_clusters_2d()
        
        # Phân tích các cụm
        cluster_features, data_with_clusters = self.analyze_clusters()
        
        # So sánh t-SNE và PCA
        comparison = self.compare_tsne_pca()
        
        print("Hoàn thành phân tích!")
        
        return {
            'cluster_features': cluster_features,
            'data_with_clusters': data_with_clusters,
            'comparison': comparison,
            'optimal_clusters': self.optimal_clusters
        }


if __name__ == "__main__":
    # Ví dụ sử dụng
    data_path = "../processed_customer_data.csv"
    clustering = CustomerClustering(data_path=data_path)
    results = clustering.run_complete_analysis()
