"""
PJM Data Analyzer - Chuyên dụng cho dữ liệu PJM hourly energy consumption
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class PJMDataAnalyzer:
    """
    Lớp phân tích dữ liệu PJM hourly energy consumption
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Khởi tạo PJM Data Analyzer
        
        Args:
            data_dir: Thư mục chứa dữ liệu PJM
        """
        self.data_dir = Path(data_dir)
        self.available_regions = {}
        self.current_data = None
        self.current_region = None
        
        # Tự động phát hiện các file dữ liệu
        self._detect_available_regions()
    
    def _detect_available_regions(self) -> None:
        """
        Tự động phát hiện các khu vực có sẵn từ thư mục dữ liệu
        """
        if not self.data_dir.exists():
            print(f"Thư mục {self.data_dir} không tồn tại!")
            return
        
        csv_files = list(self.data_dir.glob("*.csv"))
        
        for file in csv_files:
            file_name = file.stem
            if "_hourly" in file_name:
                region = file_name.replace("_hourly", "")
                self.available_regions[region] = file
                
        print(f"Phát hiện {len(self.available_regions)} khu vực:")
        for region, file_path in self.available_regions.items():
            print(f"  - {region}: {file_path.name}")
    
    def get_available_regions(self) -> List[str]:
        """
        Lấy danh sách các khu vực có sẵn
        
        Returns:
            Danh sách tên khu vực
        """
        return list(self.available_regions.keys())
    
    def load_region_data(self, region: str) -> pd.DataFrame:
        """
        Tải dữ liệu cho một khu vực cụ thể
        
        Args:
            region: Tên khu vực (ví dụ: 'PJME', 'AEP', 'COMED')
            
        Returns:
            DataFrame chứa dữ liệu đã được xử lý
        """
        if region not in self.available_regions:
            raise ValueError(f"Khu vực '{region}' không có sẵn. Các khu vực có sẵn: {self.get_available_regions()}")
        
        file_path = self.available_regions[region]
        
        # Tải dữ liệu
        print(f"Đang tải dữ liệu cho khu vực {region}...")
        data = pd.read_csv(file_path)
        
        # Tự động phát hiện tên cột
        datetime_col = data.columns[0]
        target_col = data.columns[1]
        
        # Đổi tên cột cho nhất quán
        data.columns = ['Datetime', 'MW']
        
        # Chuyển đổi datetime và sắp xếp
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.sort_values('Datetime').reset_index(drop=True)
        
        # Đặt datetime làm index
        data.set_index('Datetime', inplace=True)
        
        # Xử lý missing values
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            print(f"Phát hiện {missing_count} giá trị thiếu, đang xử lý...")
            data = data.interpolate(method='time')
        
        # Lưu trữ dữ liệu hiện tại
        self.current_data = data
        self.current_region = region
        
        print(f"Đã tải dữ liệu {region}:")
        print(f"  - Shape: {data.shape}")
        print(f"  - Khoảng thời gian: {data.index.min()} đến {data.index.max()}")
        print(f"  - Cột gốc: {target_col}")
        
        return data
    
    def get_data_statistics(self, region: str = None) -> Dict[str, any]:
        """
        Lấy thống kê cơ bản của dữ liệu
        
        Args:
            region: Tên khu vực (nếu None, dùng dữ liệu hiện tại)
            
        Returns:
            Dictionary chứa thống kê
        """
        if region:
            data = self.load_region_data(region)
        else:
            data = self.current_data
            
        if data is None:
            raise ValueError("Không có dữ liệu nào được tải. Hãy gọi load_region_data() trước.")
        
        stats = {
            'region': region or self.current_region,
            'total_records': len(data),
            'date_range': {
                'start': data.index.min(),
                'end': data.index.max(),
                'duration_years': (data.index.max() - data.index.min()).days / 365.25
            },
            'mw_statistics': data['MW'].describe().to_dict(),
            'missing_values': data.isnull().sum().sum(),
            'duplicates': data.index.duplicated().sum()
        }
        
        return stats
    
    def create_comprehensive_visualizations(self, region: str = None, save_plots: bool = False, plot_dir: str = 'plots') -> Dict[str, any]:
        """
        Tạo các biểu đồ phân tích toàn diện
        
        Args:
            region: Tên khu vực
            save_plots: Có lưu biểu đồ không
            plot_dir: Thư mục lưu biểu đồ
            
        Returns:
            Dictionary chứa các biểu đồ
        """
        if region:
            data = self.load_region_data(region)
        else:
            data = self.current_data
            region = self.current_region
            
        if data is None:
            raise ValueError("Không có dữ liệu nào được tải.")
        
        plots = {}
        
        # Tạo thư mục lưu biểu đồ
        if save_plots:
            Path(plot_dir).mkdir(exist_ok=True)
        
        # 1. Biểu đồ chuỗi thời gian tổng quan
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Lấy mẫu dữ liệu cho hiển thị (nếu dữ liệu quá lớn)
        if len(data) > 50000:
            sample_data = data.resample('D').mean()  # Lấy mẫu theo ngày
            ax.plot(sample_data.index, sample_data['MW'], alpha=0.7, linewidth=1)
            ax.set_title(f'{region} - Mức tiêu thụ điện theo thời gian (Trung bình theo ngày)')
        else:
            ax.plot(data.index, data['MW'], alpha=0.7, linewidth=0.5)
            ax.set_title(f'{region} - Mức tiêu thụ điện theo thời gian')
        
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Mức tiêu thụ (MW)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{plot_dir}/{region}_time_series.png", dpi=300, bbox_inches='tight')
        
        plots['time_series'] = fig
        plt.show()
        
        # 2. Phân tích theo mùa (nếu có đủ dữ liệu)
        if len(data) > 8760:  # Ít nhất 1 năm dữ liệu
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Theo tháng
            monthly_avg = data.groupby(data.index.month)['MW'].mean()
            axes[0, 0].bar(monthly_avg.index, monthly_avg.values)
            axes[0, 0].set_title('Trung bình tiêu thụ theo tháng')
            axes[0, 0].set_xlabel('Tháng')
            axes[0, 0].set_ylabel('MW')
            
            # Theo ngày trong tuần
            daily_avg = data.groupby(data.index.dayofweek)['MW'].mean()
            day_names = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'CN']
            axes[0, 1].bar(range(7), daily_avg.values)
            axes[0, 1].set_title('Trung bình tiêu thụ theo ngày trong tuần')
            axes[0, 1].set_xlabel('Ngày')
            axes[0, 1].set_ylabel('MW')
            axes[0, 1].set_xticks(range(7))
            axes[0, 1].set_xticklabels(day_names)
            
            # Theo giờ trong ngày
            hourly_avg = data.groupby(data.index.hour)['MW'].mean()
            axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o')
            axes[1, 0].set_title('Trung bình tiêu thụ theo giờ trong ngày')
            axes[1, 0].set_xlabel('Giờ')
            axes[1, 0].set_ylabel('MW')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Heatmap theo giờ và ngày trong tuần
            pivot_data = data.pivot_table(values='MW', index=data.index.hour, columns=data.index.dayofweek, aggfunc='mean')
            sns.heatmap(pivot_data, ax=axes[1, 1], cmap='YlOrRd', cbar_kws={'label': 'MW'})
            axes[1, 1].set_title('Heatmap tiêu thụ theo giờ và ngày')
            axes[1, 1].set_xlabel('Ngày trong tuần')
            axes[1, 1].set_ylabel('Giờ')
            axes[1, 1].set_xticklabels(day_names)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f"{plot_dir}/{region}_seasonal_analysis.png", dpi=300, bbox_inches='tight')
            
            plots['seasonal'] = fig
            plt.show()
        
        # 3. Phân phối và outliers
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Histogram
        axes[0].hist(data['MW'], bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_title('Phân phối mức tiêu thụ')
        axes[0].set_xlabel('MW')
        axes[0].set_ylabel('Tần suất')
        
        # Box plot
        axes[1].boxplot(data['MW'])
        axes[1].set_title('Box plot - Phát hiện outliers')
        axes[1].set_ylabel('MW')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(data['MW'], dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot - Kiểm tra phân phối chuẩn')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{plot_dir}/{region}_distribution_analysis.png", dpi=300, bbox_inches='tight')
        
        plots['distribution'] = fig
        plt.show()
        
        # 4. Xu hướng và thống kê cuộn
        if len(data) > 8760:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # Xu hướng theo năm
            yearly_data = data.resample('Y').mean()
            axes[0].plot(yearly_data.index, yearly_data['MW'], marker='o', linewidth=2)
            axes[0].set_title('Xu hướng trung bình hàng năm')
            axes[0].set_xlabel('Năm')
            axes[0].set_ylabel('MW')
            axes[0].grid(True, alpha=0.3)
            
            # Thống kê cuộn (rolling statistics)
            rolling_mean = data['MW'].rolling(window=24*7).mean()  # 7 ngày
            rolling_std = data['MW'].rolling(window=24*7).std()
            
            sample_data = data.resample('D').mean()
            sample_rolling_mean = rolling_mean.resample('D').mean()
            sample_rolling_std = rolling_std.resample('D').mean()
            
            axes[1].plot(sample_data.index, sample_data['MW'], alpha=0.3, label='Dữ liệu gốc')
            axes[1].plot(sample_rolling_mean.index, sample_rolling_mean.values, 'r-', label='Trung bình cuộn (7 ngày)')
            axes[1].fill_between(sample_rolling_mean.index, 
                               sample_rolling_mean.values - sample_rolling_std.values,
                               sample_rolling_mean.values + sample_rolling_std.values,
                               alpha=0.2, color='red', label='± 1 độ lệch chuẩn')
            axes[1].set_title('Thống kê cuộn')
            axes[1].set_xlabel('Thời gian')
            axes[1].set_ylabel('MW')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f"{plot_dir}/{region}_trend_analysis.png", dpi=300, bbox_inches='tight')
            
            plots['trend'] = fig
            plt.show()
        
        return plots
    
    def compare_regions(self, regions: List[str], save_plots: bool = False, plot_dir: str = 'plots') -> Dict[str, any]:
        """
        So sánh nhiều khu vực
        
        Args:
            regions: Danh sách khu vực cần so sánh
            save_plots: Có lưu biểu đồ không
            plot_dir: Thư mục lưu biểu đồ
            
        Returns:
            Dictionary chứa kết quả so sánh
        """
        comparison_data = {}
        
        # Tải dữ liệu cho tất cả các khu vực
        for region in regions:
            if region not in self.available_regions:
                print(f"Bỏ qua khu vực {region} - không có dữ liệu")
                continue
            
            data = self.load_region_data(region)
            comparison_data[region] = data
        
        if not comparison_data:
            raise ValueError("Không có dữ liệu nào để so sánh")
        
        # Tạo biểu đồ so sánh
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # So sánh xu hướng
        for region, data in comparison_data.items():
            # Lấy mẫu dữ liệu theo tuần
            weekly_data = data.resample('W').mean()
            axes[0, 0].plot(weekly_data.index, weekly_data['MW'], alpha=0.7, label=region)
        
        axes[0, 0].set_title('So sánh xu hướng theo thời gian')
        axes[0, 0].set_xlabel('Thời gian')
        axes[0, 0].set_ylabel('MW')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # So sánh thống kê
        stats_data = []
        for region, data in comparison_data.items():
            stats_data.append({
                'Region': region,
                'Mean': data['MW'].mean(),
                'Median': data['MW'].median(),
                'Std': data['MW'].std(),
                'Min': data['MW'].min(),
                'Max': data['MW'].max()
            })
        
        stats_df = pd.DataFrame(stats_data)
        x_pos = np.arange(len(stats_df))
        
        axes[0, 1].bar(x_pos, stats_df['Mean'], alpha=0.7)
        axes[0, 1].set_title('So sánh mức tiêu thụ trung bình')
        axes[0, 1].set_xlabel('Khu vực')
        axes[0, 1].set_ylabel('MW')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(stats_df['Region'])
        
        # So sánh phân phối
        mw_data = [data['MW'].values for data in comparison_data.values()]
        axes[1, 0].boxplot(mw_data, labels=list(comparison_data.keys()))
        axes[1, 0].set_title('So sánh phân phối mức tiêu thụ')
        axes[1, 0].set_xlabel('Khu vực')
        axes[1, 0].set_ylabel('MW')
        
        # So sánh pattern theo giờ
        for region, data in comparison_data.items():
            hourly_avg = data.groupby(data.index.hour)['MW'].mean()
            axes[1, 1].plot(hourly_avg.index, hourly_avg.values, marker='o', label=region)
        
        axes[1, 1].set_title('So sánh pattern theo giờ trong ngày')
        axes[1, 1].set_xlabel('Giờ')
        axes[1, 1].set_ylabel('MW')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            Path(plot_dir).mkdir(exist_ok=True)
            region_names = "_".join(regions)
            plt.savefig(f"{plot_dir}/comparison_{region_names}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'comparison_data': comparison_data,
            'statistics': stats_df,
            'plot': fig
        }
    
    def prepare_for_modeling(self, region: str, normalize: bool = True) -> Tuple[pd.DataFrame, any]:
        """
        Chuẩn bị dữ liệu cho modeling
        
        Args:
            region: Tên khu vực
            normalize: Có chuẩn hóa dữ liệu không
            
        Returns:
            Tuple (processed_data, scaler)
        """
        data = self.load_region_data(region)
        
        processed_data = data.copy()
        scaler = None
        
        if normalize:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            processed_data['MW'] = scaler.fit_transform(processed_data[['MW']])
            print("Đã chuẩn hóa dữ liệu sử dụng MinMaxScaler")
        
        return processed_data, scaler
    
    def generate_region_report(self, region: str, save_path: str = None) -> str:
        """
        Tạo báo cáo chi tiết cho một khu vực
        
        Args:
            region: Tên khu vực
            save_path: Đường dẫn lưu báo cáo
            
        Returns:
            Nội dung báo cáo
        """
        stats = self.get_data_statistics(region)
        
        report = f"""
BÁOCÁO PHÂN TÍCH DỮ LIỆU KHU VỰC {region}
{'='*50}

1. THÔNG TIN TỔNG QUAN:
   - Tổng số bản ghi: {stats['total_records']:,}
   - Khoảng thời gian: {stats['date_range']['start']} đến {stats['date_range']['end']}
   - Thời gian thu thập: {stats['date_range']['duration_years']:.1f} năm
   - Giá trị thiếu: {stats['missing_values']}
   - Bản ghi trùng lặp: {stats['duplicates']}

2. THỐNG KÊ MỨC TIÊU THỤ ĐIỆN (MW):
   - Trung bình: {stats['mw_statistics']['mean']:.2f} MW
   - Trung vị: {stats['mw_statistics']['50%']:.2f} MW
   - Độ lệch chuẩn: {stats['mw_statistics']['std']:.2f} MW
   - Giá trị nhỏ nhất: {stats['mw_statistics']['min']:.2f} MW
   - Giá trị lớn nhất: {stats['mw_statistics']['max']:.2f} MW
   - Tứ phân vị 25%: {stats['mw_statistics']['25%']:.2f} MW
   - Tứ phân vị 75%: {stats['mw_statistics']['75%']:.2f} MW

3. CHẤT LƯỢNG DỮ LIỆU:
   - Độ hoàn chỉnh: {((stats['total_records'] - stats['missing_values']) / stats['total_records'] * 100):.2f}%
   - Không có bản ghi trùng lặp: {"Có" if stats['duplicates'] == 0 else "Không"}

4. ĐÁNH GIÁ:
   - Dữ liệu phù hợp cho time series forecasting
   - Cần chuẩn hóa trước khi modeling
   - Có thể sử dụng window size 24h cho daily pattern
        """
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Báo cáo đã được lưu tại: {save_path}")
        
        return report.strip() 