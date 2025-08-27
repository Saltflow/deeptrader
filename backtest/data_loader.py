# coding=utf-8
# 数据加载模块 - 基于AkShare离线数据

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class BacktestDataLoader:
    """回测数据加载器，专门用于加载离线数据"""
    
    def __init__(self, data_dir: str = "data/akshare"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        
        # 支持的交易所
        self.exchange_mapping = {
            'SHFE': '上海期货交易所',
            'DCE': '大连商品交易所', 
            'CZCE': '郑州商品交易所',
            'CFFEX': '中国金融期货交易所',
            'GFEX': '广州期货交易所',
            'INE': '上海国际能源交易中心'
        }
    
    def load_minute_data(self, symbol: str, period: str = "60") -> Optional[pd.DataFrame]:
        """
        加载分钟级别数据
        
        Args:
            symbol: 合约代码 (如: rb2410, RB0)
            period: 数据周期 (60, 30, 15, 5, 1)
            
        Returns:
            DataFrame包含OHLCV数据，或None如果数据不存在
        """
        filename = f"minute_{symbol}_{period}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"数据文件不存在: {filepath}")
            return None
        
        try:
            data = pd.read_csv(filepath)
            
            # 数据清洗和转换
            if 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index('datetime', inplace=True)
            
            # 确保列名标准化
            column_mapping = {
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'hold': 'open_interest'
            }
            
            data.rename(columns=column_mapping, inplace=True)
            
            # 确保必要的列存在
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"缺少必要列: {col}")
                    return None
            
            # 排序并去除重复
            data = data.sort_index()
            data = data[~data.index.duplicated(keep='first')]
            
            logger.info(f"成功加载 {symbol} 的{period}分钟数据，共 {len(data)} 条记录")
            return data
            
        except Exception as e:
            logger.error(f"加载数据失败 {filepath}: {e}")
            return None
    
    def load_multiple_symbols(self, symbols: List[str], period: str = "60") -> Dict[str, pd.DataFrame]:
        """
        批量加载多个合约数据
        
        Args:
            symbols: 合约代码列表
            period: 数据周期
            
        Returns:
            字典，键为合约代码，值为对应的DataFrame
        """
        results = {}
        for symbol in symbols:
            data = self.load_minute_data(symbol, period)
            if data is not None:
                results[symbol] = data
        return results
    
    def get_available_symbols(self, period: str = "60") -> List[str]:
        """
        获取可用的合约代码列表
        
        Args:
            period: 数据周期
            
        Returns:
            可用合约代码列表
        """
        symbols = []
        pattern = f"minute_*_{period}.csv"
        
        for filepath in self.data_dir.glob(pattern):
            filename = filepath.name
            # 提取合约代码: minute_{symbol}_{period}.csv
            parts = filename.split('_')
            if len(parts) >= 3:
                symbol = parts[1]
                symbols.append(symbol)
        
        return sorted(set(symbols))
    
    def get_data_summary(self, symbol: str, period: str = "60") -> Dict:
        """
        获取数据摘要信息
        
        Args:
            symbol: 合约代码
            period: 数据周期
            
        Returns:
            数据摘要字典
        """
        data = self.load_minute_data(symbol, period)
        if data is None:
            return {}
        
        return {
            'symbol': symbol,
            'period': period,
            'start_date': data.index.min(),
            'end_date': data.index.max(),
            'total_records': len(data),
            'columns': list(data.columns),
            'price_range': {
                'min': data['low'].min(),
                'max': data['high'].max(),
                'avg': data['close'].mean()
            },
            'volume_stats': {
                'total': data['volume'].sum(),
                'avg': data['volume'].mean(),
                'max': data['volume'].max()
            }
        }
    
    def resample_data(self, data: pd.DataFrame, new_period: str) -> pd.DataFrame:
        """
        重采样数据到不同的时间周期
        
        Args:
            data: 原始数据DataFrame
            new_period: 新周期 ('1H', '4H', '1D', etc.)
            
        Returns:
            重采样后的DataFrame
        """
        if data.empty:
            return data
        
        # OHLC重采样规则
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum'
        }
        
        # 如果存在持仓量数据，使用最后值
        if 'open_interest' in data.columns:
            ohlc_dict['open_interest'] = 'last'
        
        try:
            resampled = data.resample(new_period).apply(ohlc_dict)
            resampled.dropna(inplace=True)
            return resampled
        except Exception as e:
            logger.error(f"重采样数据失败: {e}")
            return data


def create_test_data() -> pd.DataFrame:
    """创建测试用的模拟数据"""
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='H')
    n = len(dates)
    
    # 生成模拟价格数据
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, n)
    prices = 1000 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n))),
        'close': prices,
        'volume': np.random.lognormal(8, 1.5, n),
        'open_interest': np.random.randint(10000, 50000, n)
    }, index=dates)
    
    return data