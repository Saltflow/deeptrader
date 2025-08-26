# coding=utf-8
#
# Copyright 2016 timercrack
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
import akshare as ak
import pandas as pd
import datetime
import os
import logging
import json
from typing import Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class AkshareDataLoader:
    """基于AkShare的数据加载器，支持自动持久化"""
    
    def __init__(self, data_dir: str = "data/akshare"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 交易所映射
        self.exchange_mapping = {
            'SHFE': '上海期货交易所',
            'DCE': '大连商品交易所', 
            'CZCE': '郑州商品交易所',
            'CFFEX': '中国金融期货交易所',
            'GFEX': '广州期货交易所',
            'INE': '上海国际能源交易中心'
        }
        
        # 数据缓存
        self._cache = {}
    
    def _get_cache_key(self, data_type: str, symbol: str, period: str = None) -> str:
        """生成缓存键"""
        if period:
            return f"{data_type}_{symbol}_{period}"
        return f"{data_type}_{symbol}"
    
    def _save_to_cache(self, key: str, data: pd.DataFrame):
        """保存数据到缓存"""
        self._cache[key] = data
    
    def _load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        return self._cache.get(key)
    
    def _get_data_path(self, data_type: str, symbol: str, period: str = None) -> Path:
        """获取数据文件路径"""
        if period:
            filename = f"{data_type}_{symbol}_{period}.csv"
        else:
            filename = f"{data_type}_{symbol}.csv"
        return self.data_dir / filename
    
    def _save_data(self, data: pd.DataFrame, filepath: Path):
        """保存数据到文件"""
        try:
            data.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"数据已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
    
    def _load_data(self, filepath: Path) -> Optional[pd.DataFrame]:
        """从文件加载数据"""
        if filepath.exists():
            try:
                data = pd.read_csv(filepath)
                # 转换时间列
                if 'datetime' in data.columns:
                    data['datetime'] = pd.to_datetime(data['datetime'])
                logger.info(f"从文件加载数据: {filepath}")
                return data
            except Exception as e:
                logger.error(f"加载数据文件失败: {e}")
        return None
    
    def get_future_minute_data(self, symbol: str, period: str = "60", 
                              force_update: bool = False) -> Optional[pd.DataFrame]:
        """
        获取期货分钟级别数据
        
        Args:
            symbol: 合约代码
            period: 周期 (1, 5, 15, 30, 60)
            force_update: 是否强制更新数据
            
        Returns:
            DataFrame包含分钟数据
        """
        cache_key = self._get_cache_key('minute', symbol, period)
        filepath = self._get_data_path('minute', symbol, period)
        
        # 检查缓存
        if not force_update:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # 检查本地文件
        if not force_update and filepath.exists():
            local_data = self._load_data(filepath)
            if local_data is not None:
                self._save_to_cache(cache_key, local_data)
                return local_data
        
        # 从网络获取数据
        try:
            logger.info(f"正在获取合约 {symbol} 的{period}分钟数据...")
            data = ak.futures_zh_minute_sina(symbol=symbol, period=period)
            
            if data is not None and len(data) > 0:
                # 保存到缓存和文件
                self._save_to_cache(cache_key, data)
                self._save_data(data, filepath)
                
                logger.info(f"成功获取 {len(data)} 条{period}分钟数据")
                return data
            else:
                logger.warning(f"未获取到 {symbol} 的{period}分钟数据")
                return None
                
        except Exception as e:
            logger.error(f"获取{period}分钟数据失败: {e}")
            return None
    
    def get_future_daily_data(self, symbol: str, start_date: str = None, 
                             end_date: str = None, force_update: bool = False) -> Optional[pd.DataFrame]:
        """
        获取期货日线数据
        
        Args:
            symbol: 合约代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            force_update: 是否强制更新数据
            
        Returns:
            DataFrame包含日线数据
        """
        cache_key = self._get_cache_key('daily', symbol)
        filepath = self._get_data_path('daily', symbol)
        
        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y%m%d')
        
        # 检查缓存
        if not force_update:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # 检查本地文件
        if not force_update and filepath.exists():
            local_data = self._load_data(filepath)
            if local_data is not None:
                # 检查是否需要更新
                if not self._need_update(local_data, end_date):
                    self._save_to_cache(cache_key, local_data)
                    return local_data
        
        # 从网络获取数据
        try:
            logger.info(f"正在获取合约 {symbol} 的日线数据 ({start_date} 到 {end_date})...")
            data = ak.futures_zh_daily(symbol=symbol, start_date=start_date, end_date=end_date)
            
            if data is not None and len(data) > 0:
                # 保存到缓存和文件
                self._save_to_cache(cache_key, data)
                self._save_data(data, filepath)
                
                logger.info(f"成功获取 {len(data)} 条日线数据")
                return data
            else:
                logger.warning(f"未获取到 {symbol} 的日线数据")
                return None
                
        except Exception as e:
            logger.error(f"获取日线数据失败: {e}")
            return None
    
    def _need_update(self, data: pd.DataFrame, end_date: str) -> bool:
        """检查数据是否需要更新"""
        if 'date' in data.columns and len(data) > 0:
            last_date = pd.to_datetime(data['date'].iloc[-1])
            current_date = pd.to_datetime(end_date)
            return last_date < current_date
        return True
    
    def get_contract_list(self, exchange: str = None) -> Optional[pd.DataFrame]:
        """获取合约列表"""
        try:
            contracts = ak.futures_display_main_sina()
            if exchange:
                contracts = contracts[contracts['exchange'] == exchange]
            return contracts
        except Exception as e:
            logger.error(f"获取合约列表失败: {e}")
            return None
    
    def get_main_contracts(self) -> Optional[pd.DataFrame]:
        """获取主力合约"""
        try:
            return ak.futures_main_sina()
        except Exception as e:
            logger.error(f"获取主力合约失败: {e}")
            return None
    
    def get_market_overview(self) -> Dict:
        """获取市场概览"""
        overview = {}
        contracts = self.get_contract_list()
        
        if contracts is not None:
            for exchange in self.exchange_mapping.keys():
                exchange_contracts = contracts[contracts['exchange'] == exchange]
                overview[exchange] = {
                    'contract_count': len(exchange_contracts),
                    'chinese_name': self.exchange_mapping.get(exchange, exchange),
                    'sample_symbols': exchange_contracts['symbol'].head(5).tolist()
                }
        
        return overview
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("缓存已清空")


def main():
    """主函数示例"""
    import argparse
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='AkShare数据加载器')
    parser.add_argument('--symbol', '-s', default='rb2410', help='合约代码')
    parser.add_argument('--period', '-p', default='60', choices=['1', '5', '15', '30', '60'], help='数据周期')
    parser.add_argument('--data-dir', '-d', default='data/akshare', help='数据存储目录')
    parser.add_argument('--force', '-f', action='store_true', help='强制更新数据')
    parser.add_argument('--overview', '-o', action='store_true', help='显示市场概览')
    
    args = parser.parse_args()
    
    loader = AkshareDataLoader(data_dir=args.data_dir)
    
    if args.overview:
        print("=== 期货市场概览 ===")
        overview = loader.get_market_overview()
        for exchange, info in overview.items():
            print(f"{exchange} ({info['chinese_name']}): {info['contract_count']}个合约")
            print(f"  示例合约: {info['sample_symbols']}")
    else:
        print(f"=== 获取合约 {args.symbol} 的{args.period}分钟数据 ===")
        data = loader.get_future_minute_data(args.symbol, args.period, args.force)
        
        if data is not None:
            print(f"成功获取 {len(data)} 条数据")
            print(f"时间范围: {data['datetime'].min()} 到 {data['datetime'].max()}")
            print("\n最新5条数据:")
            print(data.tail())
            
            # 显示数据文件位置
            filepath = loader._get_data_path('minute', args.symbol, args.period)
            print(f"\n数据文件: {filepath}")


if __name__ == "__main__":
    main()