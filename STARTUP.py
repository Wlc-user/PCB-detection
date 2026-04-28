"""
3C电子产品降价预测系统
基于供应链传导机制的量化分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 统计分析
from scipy import stats
from scipy.signal import correlate
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


class SupplyChainAnalyzer:
    """供应链传导分析器 - 基于Granger因果检验和CCF分析"""
    
    def __init__(self):
        self.data = {}
        self.results = {}
        
    def generate_sample_data(self, periods: int = 25) -> pd.DataFrame:
        """生成模拟数据（基于真实趋势）"""
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='M')
        
        np.random.seed(42)
        
        # L0: AI算力需求 (趋势上升)
        ai_demand = 50 + np.arange(periods) * 2 + np.random.normal(0, 5, periods)
        
        # L1: DRAM价格 (滞后AI需求4个月)
        dram_price = 40 + np.concatenate([np.zeros(4), ai_demand[:-4] * 0.3]) + np.random.normal(0, 3, periods)
        dram_price = np.maximum(dram_price, 20)
        
        # 面板价格
        panel_price = 60 + np.sin(np.arange(periods) * 0.5) * 10 + np.random.normal(0, 4, periods)
        
        # L2: 渠道库存DSI (受DRAM价格影响，滞后2个月)
        inventory_dsi = 50 + np.concatenate([np.zeros(2), dram_price[:-2] * 0.4]) + np.random.normal(0, 5, periods)
        
        # L3: 出货量 (受库存影响，滞后1个月)
        shipment = 100 - np.concatenate([np.zeros(1), (inventory_dsi[:-1] - 50) * 0.8]) + np.random.normal(0, 8, periods)
        shipment = np.maximum(shipment, 30)
        
        # L4: 终端价格 (安泰股价/售价)
        terminal_price = 80 + np.concatenate([np.zeros(3), dram_price[:-3] * 0.3]) + np.random.normal(0, 3, periods)
        
        # L5: 产业链股价 (存储股超额收益)
        stock_return = 5 + np.concatenate([np.zeros(2), dram_price[:-2] * 0.2]) + np.random.normal(0, 4, periods)
        
        # L6: 降价概率 (综合多个因素)
        discount_prob = 30 + (dram_price * -0.3) + (inventory_dsi * 0.2) + (shipment * -0.1) + np.random.normal(0, 8, periods)
        discount_prob = np.clip(discount_prob, 0, 100)
        
        df = pd.DataFrame({
            'date': dates,
            'ai_demand': ai_demand,                    # L0: AI算力需求
            'dram_price': dram_price,                  # L1: DRAM价格指数
            'panel_price': panel_price,                # L1: 面板价格
            'inventory_dsi': inventory_dsi,            # L2: 渠道库存天数
            'shipment': shipment,                      # L3: 出货量
            'terminal_price': terminal_price,          # L4: 终端定价
            'stock_return': stock_return,              # L5: 产业链股价
            'discount_prob': discount_prob             # 降价概率
        })
        
        return df
    
    def granger_test(self, df: pd.DataFrame, var1: str, var2: str, max_lag: int = 4) -> Dict:
        """
        Granger因果检验
        检验 var2 是否是 var1 的格兰杰原因
        """
        data = pd.DataFrame({var1: df[var1], var2: df[var2]}).dropna()
        
        results = {}
        best_lag = None
        best_pvalue = 1.0
        
        for lag in range(1, max_lag + 1):
            try:
                test_result = grangercausalitytests(data, maxlag=lag, verbose=False)
                pvalue = test_result[lag][0]['ssr_ftest'][1]
                results[lag] = pvalue
                
                if pvalue < best_pvalue:
                    best_pvalue = pvalue
                    best_lag = lag
            except:
                continue
        
        return {
            'optimal_lag': best_lag,
            'best_pvalue': best_pvalue,
            'is_significant': best_pvalue < 0.05,
            'all_results': results
        }
    
    def cross_correlation_analysis(self, df: pd.DataFrame, var1: str, var2: str, max_lag: int = 6) -> Dict:
        """
        互相关分析(CCF)
        找出两个时间序列之间的最优滞后相关性
        """
        # 标准化
        x = (df[var1] - df[var1].mean()) / df[var1].std()
        y = (df[var2] - df[var2].mean()) / df[var2].std()
        
        # 计算互相关
        correlations = []
        lags = range(-max_lag, max_lag + 1)
        
        for lag in lags:
            if lag < 0:
                corr = x.iloc[-lag:].corr(y.iloc[:lag])
            elif lag > 0:
                corr = x.iloc[:-lag].corr(y.iloc[lag:])
            else:
                corr = x.corr(y)
            correlations.append(corr if not np.isnan(corr) else 0)
        
        # 找出最优滞后
        best_idx = np.argmax(np.abs(correlations))
        best_lag = lags[best_idx]
        best_corr = correlations[best_idx]
        
        return {
            'optimal_lag': best_lag,
            'max_correlation': best_corr,
            'lags': list(lags),
            'correlations': correlations
        }
    
    def analyze_transmission_chain(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析完整传导链"""
        
        # 定义传导关系
        pairs = [
            ('ai_demand', 'dram_price', 4, 'AI算力需求 → DRAM价格'),
            ('dram_price', 'inventory_dsi', 2, 'DRAM涨价 → 渠道库存DSI'),
            ('dram_price', 'terminal_price', 3, 'DRAM涨价 → 终端定价'),
            ('inventory_dsi', 'shipment', 1, '库存DSI → 出货量'),
            ('dram_price', 'stock_return', 2, 'DRAM涨价 → 存储股收益'),
            ('terminal_price', 'discount_prob', 1, '终端价格 → 降价概率')
        ]
        
        results = []
        
        for var1, var2, expected_lag, mechanism in pairs:
            # Granger因果检验
            granger_res = self.granger_test(df, var1, var2, max_lag=6)
            
            # 互相关分析
            ccf_res = self.cross_correlation_analysis(df, var1, var2, max_lag=6)
            
            # 判断显著性
            is_significant = granger_res['is_significant'] or abs(ccf_res['max_correlation']) > 0.5
            
            results.append({
                '传导层级': f'{var1} → {var2}',
                '机制解读': mechanism,
                '最优滞后': f"{ccf_res['optimal_lag']}个月",
                '最大相关系数': round(ccf_res['max_correlation'], 3),
                'Granger p值': round(granger_res['best_pvalue'], 4),
                '显著性': '显著★' if is_significant else '不显著'
            })
        
        return pd.DataFrame(results)


class PricePredictionModel:
    """降价预测模型"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scaler = StandardScaler()
        self.feature_weights = {
            'dram_price': -0.25,      # DRAM价格上涨 → 降价概率下降
            'inventory_dsi': 0.30,     # 库存高 → 降价概率上升
            'shipment': -0.20,         # 出货量下降 → 降价概率上升
            'terminal_price': -0.15,   # 终端价格上涨 → 降价概率下降
            'stock_return': 0.10       # 股价表现 → 信号强度
        }
    
    def compute_zscore(self, series: pd.Series, window: int = 6) -> pd.Series:
        """计算Z-Score (异常检测)"""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        zscore = (series - rolling_mean) / rolling_std
        return zscore.fillna(0)
    
    def compute_discount_probability(self) -> pd.DataFrame:
        """计算降价概率"""
        
        # 计算各个维度的Z-Score
        zscore_features = {}
        for feature in self.feature_weights.keys():
            if feature in self.df.columns:
                zscore_features[feature] = self.compute_zscore(self.df[feature])
        
        # 加权计算综合得分
        composite_score = np.zeros(len(self.df))
        for feature, weight in self.feature_weights.items():
            if feature in zscore_features:
                composite_score += weight * zscore_features[feature].fillna(0)
        
        # 转换为概率
        discount_prob = 50 + composite_score * 15
        discount_prob = np.clip(discount_prob, 0, 100)
        
        # 触发条件核查
        trigger_conditions = self.check_trigger_conditions(zscore_features)
        
        result = pd.DataFrame({
            'date': self.df['date'],
            'discount_probability': discount_prob,
            'signal_strength': composite_score,
            'alert_level': self.get_alert_level(discount_prob)
        })
        
        # 添加各维度Z-Score
        for feature, zscore in zscore_features.items():
            result[f'zscore_{feature}'] = zscore
        
        # 添加触发条件
        result['trigger_score'] = trigger_conditions['total_score']
        result['trigger_details'] = trigger_conditions['details']
        
        return result
    
    def check_trigger_conditions(self, zscore_features: Dict) -> Dict:
        """检查触发条件 (类似T1/T2信号)"""
        
        current_zscore = {k: v.iloc[-1] if len(v) > 0 else 0 
                          for k, v in zscore_features.items()}
        
        conditions = []
        total_score = 0
        
        # T1: 库存信号
        inventory_z = current_zscore.get('inventory_dsi', 0)
        if inventory_z > 2.0:
            conditions.append(f"库存高企(+{inventory_z:.2f}σ)")
            total_score += 2
        elif inventory_z > 1.0:
            conditions.append(f"库存偏高(+{inventory_z:.2f}σ)")
            total_score += 1
        
        # T2: 出货信号
        shipment_z = current_zscore.get('shipment', 0)
        if shipment_z < -1.5:
            conditions.append(f"出货疲软({shipment_z:.2f}σ)")
            total_score += 2
        elif shipment_z < -0.8:
            conditions.append(f"出货偏弱({shipment_z:.2f}σ)")
            total_score += 1
        
        # DRAM价格信号
        dram_z = current_zscore.get('dram_price', 0)
        if dram_z > 1.5:
            conditions.append(f"DRAM涨价(+{dram_z:.2f}σ)")
            total_score -= 1
        
        # 终端价格信号
        terminal_z = current_zscore.get('terminal_price', 0)
        if terminal_z < -1.0:
            conditions.append(f"终端降价({terminal_z:.2f}σ)")
            total_score += 1
        
        return {
            'total_score': total_score,
            'details': '; '.join(conditions) if conditions else '无明显触发信号',
            'conditions_count': len(conditions)
        }
    
    def get_alert_level(self, probabilities: pd.Series) -> List[str]:
        """获取预警等级"""
        levels = []
        for p in probabilities:
            if p >= 60:
                levels.append('高度预警')
            elif p >= 40:
                levels.append('中度预警')
            elif p >= 20:
                levels.append('轻度预警')
            else:
                levels.append('信号偏弱')
        return levels
    
    def predict_future(self, months_ahead: int = 3) -> Dict:
        """预测未来降价概率"""
        
        # 基于最近趋势预测
        recent_prob = self.compute_discount_probability()
        recent_3m = recent_prob['discount_probability'].tail(3).mean()
        
        # 趋势因子
        trend = recent_prob['discount_probability'].diff().tail(3).mean()
        
        predictions = []
        current_prob = recent_3m
        
        for i in range(months_ahead):
            # 简单趋势预测
            current_prob = current_prob + trend + np.random.normal(0, 2)
            current_prob = np.clip(current_prob, 0, 100)
            predictions.append(current_prob)
        
        # 计算预测窗口
        start_date = datetime.now()
        end_date = start_date + timedelta(days=months_ahead * 30)
        
        return {
            'probability': np.mean(predictions),
            'range': (np.min(predictions), np.max(predictions)),
            'trend': '上升' if trend > 0 else '下降',
            'time_range': f"{start_date.strftime('%Y-%m')} - {end_date.strftime('%Y-%m')}",
            'months_ahead': months_ahead
        }


class VisualizationReporter:
    """可视化报告生成器"""
    
    def __init__(self):
        self.style = {
            'figure.figsize': (14, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12
        }
        plt.rcParams.update(self.style)
    
    def plot_forecast(self, df: pd.DataFrame, title: str = "降价预测趋势"):
        """绘制预测趋势图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 柱状图
        colors = ['#ff6b6b' if p > 40 else '#4ecdc4' for p in df['discount_probability']]
        bars = ax.bar(df['date'], df['discount_probability'], color=colors, alpha=0.7, label='降价概率(%)')
        
        # 折线图
        ax.plot(df['date'], df['discount_probability'], 'o-', color='#ff6b6b', linewidth=2, markersize=6)
        
        # 阈值线
        ax.axhline(y=40, color='orange', linestyle='--', linewidth=1.5, label='中度预警线(40%)')
        ax.axhline(y=60, color='red', linestyle='--', linewidth=1.5, label='高度预警线(60%)')
        
        ax.set_xlabel('日期')
        ax.set_ylabel('降价概率 (%)')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_zscore_heatmap(self, df: pd.DataFrame, features: List[str]):
        """绘制Z-Score热力图"""
        zscore_cols = [f'zscore_{f}' for f in features if f'zscore_{f}' in df.columns]
        
        if not zscore_cols:
            return None
        
        data = df[zscore_cols].T
        feature_labels = [col.replace('zscore_', '') for col in zscore_cols]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)
        
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['date'].dt.strftime('%Y-%m'), rotation=45, ha='right')
        ax.set_yticks(range(len(feature_labels)))
        ax.set_yticklabels(feature_labels)
        
        plt.colorbar(im, ax=ax, label='Z-Score')
        ax.set_title('各维度Z-Score热力图 (最近12个月)')
        
        plt.tight_layout()
        return fig
    
    def plot_transmission_chain(self, results_df: pd.DataFrame):
        """绘制传导链分析图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 定义层级位置
        layers = [
            {'y': 0.85, 'name': 'L0: AI算力需求', 'color': '#3498db'},
            {'y': 0.65, 'name': 'L1: 存储/面板成本', 'color': '#2ecc71'},
            {'y': 0.45, 'name': 'L2: 渠道库存', 'color': '#f39c12'},
            {'y': 0.25, 'name': 'L3: 出货预期', 'color': '#e74c3c'},
            {'y': 0.05, 'name': 'L4: 终端定价/降价', 'color': '#9b59b6'}
        ]
        
        # 绘制层级
        for layer in layers:
            rect = plt.Rectangle((0.1, layer['y']-0.08), 0.8, 0.12, 
                                 facecolor=layer['color'], alpha=0.3, 
                                 edgecolor=layer['color'], linewidth=2)
            ax.add_patch(rect)
            ax.text(0.5, layer['y'], layer['name'], ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        # 绘制传导箭头
        arrows = [
            (0.55, 0.82, 0.55, 0.72, '4个月'),
            (0.55, 0.62, 0.55, 0.52, '2个月'),
            (0.55, 0.42, 0.55, 0.32, '1个月'),
            (0.55, 0.22, 0.55, 0.12, '3个月')
        ]
        
        for x1, y1, x2, y2, label in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax.text((x1+x2)/2 + 0.05, (y1+y2)/2, label, fontsize=9, color='red')
        
        # 添加传导机制文本
        mechanisms = [
            "AI算力需求 → HBM涨价 → 挤占DRAM产能",
            "DRAM涨价 → 渠道库存被动累积",
            "库存高企 → 出货量下降",
            "成本压力 → 终端提价 → 降价预期"
        ]
        
        for i, text in enumerate(mechanisms):
            ax.text(0.05, 0.95 - i*0.05, f'• {text}', fontsize=9, color='#555')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('供应链传导机制与量化关系图', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig


def main():
    """主函数"""
    print("=" * 60)
    print("3C电子产品降价预测系统")
    print("基于供应链传导机制的量化分析")
    print("=" * 60)
    
    # 1. 初始化分析器
    analyzer = SupplyChainAnalyzer()
    predictor = None
    reporter = VisualizationReporter()
    
    # 2. 生成/加载数据
    print("\n[1/5] 加载供应链数据...")
    df = analyzer.generate_sample_data(periods=25)
    print(f"    数据量: {len(df)} 条月度记录")
    print(f"    时间范围: {df['date'].min().strftime('%Y-%m')} - {df['date'].max().strftime('%Y-%m')}")
    
    # 3. 传导链分析
    print("\n[2/5] 执行供应链传导分析...")
    transmission_results = analyzer.analyze_transmission_chain(df)
    print("\n传导分析结果:")
    print(transmission_results.to_string(index=False))
    
    # 4. 降价预测
    print("\n[3/5] 计算降价概率...")
    predictor = PricePredictionModel(df)
    prediction_result = predictor.compute_discount_probability()
    
    # 最新预测
    latest = prediction_result.iloc[-1]
    print(f"\n最新预测结果:")
    print(f"    降价概率: {latest['discount_probability']:.1f}%")
    print(f"    预警等级: {latest['alert_level']}")
    print(f"    触发信号: {latest['trigger_details']}")
    
    # 5. 未来预测
    print("\n[4/5] 预测未来3个月趋势...")
    future_pred = predictor.predict_future(months_ahead=3)
    print(f"\n未来预测:")
    print(f"    预测窗口: {future_pred['time_range']}")
    print(f"    预测概率: {future_pred['probability']:.1f}%")
    print(f"    预测区间: {future_pred['range'][0]:.1f}% - {future_pred['range'][1]:.1f}%")
    print(f"    趋势判断: {future_pred['trend']}")
    
    # 6. 生成报告
    print("\n[5/5] 生成可视化报告...")
    
    # 合并结果
    final_report = prediction_result.copy()
    final_report['future_probability'] = future_pred['probability']
    
    # 生成图表
    fig1 = reporter.plot_forecast(final_report.tail(12), "降价预测趋势 (近12个月)")
    fig2 = reporter.plot_zscore_heatmap(final_report, 
                                        ['dram_price', 'inventory_dsi', 'shipment', 'terminal_price'])
    fig3 = reporter.plot_transmission_chain(transmission_results)
    
    # 显示图表
    plt.show()
    
    # 输出综合判断
    print("\n" + "=" * 60)
    print("综合判断")
    print("=" * 60)
    
    prob = latest['discount_probability']
    if prob >= 40:
        print(f"⚠️  中度预警")
        print(f"触发条件: {latest['trigger_details']}")
    elif prob >= 20:
        print(f"📊 轻度预警")
        print(f"关注信号: {latest['trigger_details']}")
    else:
        print(f"✅ 信号偏弱")
    
    print(f"\n降价概率: {prob:.0f}%")
    print(f"预测窗口: {future_pred['time_range']}")
    
    # 保存报告
    final_report.to_csv('discount_prediction_report.csv', index=False, encoding='utf-8-sig')
    print(f"\n报告已保存: discount_prediction_report.csv")
    
    return final_report, transmission_results


if __name__ == '__main__':
    results_df, transmission_df = main()