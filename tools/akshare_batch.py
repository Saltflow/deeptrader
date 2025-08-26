#!/usr/bin/env python
# coding=utf-8
#
# AkShare批量数据获取工具 - 获取所有大宗商品数据
#
import argparse
import logging
import pandas as pd
import time
from typing import List, Dict, Optional
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False
    print("请安装AkShare: pip install akshare")


class AkshareBatchFetcher:
    """批量获取AkShare数据的工具"""
    
    def __init__(self, data_dir: str = "data/akshare"):
        if not HAS_AKSHARE:
            raise ImportError("AkShare未安装")
            
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
    
    def get_all_contracts(self) -> Optional[pd.DataFrame]:
        """获取所有主力合约列表"""
        try:
            contracts = ak.futures_display_main_sina()
            logging.info(f"成功获取 {len(contracts)} 个主力合约")
            return contracts
        except Exception as e:
            logging.error(f"获取合约列表失败: {e}")
            return None
    
    def get_contract_minute_data(self, symbol: str, period: str = "60", 
                               force_update: bool = False) -> Optional[pd.DataFrame]:
        """获取单个合约的分钟级别数据"""
        filepath = self.data_dir / f"minute_{symbol}_{period}.csv"
        
        # 检查本地文件
        if not force_update and filepath.exists():
            try:
                data = pd.read_csv(filepath)
                if 'datetime' in data.columns:
                    data['datetime'] = pd.to_datetime(data['datetime'])
                logging.info(f"从文件加载数据: {filepath}")
                return data
            except Exception as e:
                logging.error(f"加载数据文件失败: {e}")
        
        # 从网络获取数据
        try:
            logging.info(f"正在获取合约 {symbol} 的{period}分钟数据...")
            data = ak.futures_zh_minute_sina(symbol=symbol, period=period)
            
            if data is not None and len(data) > 0:
                # 保存到文件
                data.to_csv(filepath, index=False, encoding='utf-8-sig')
                logging.info(f"成功获取 {len(data)} 条{period}分钟数据，已保存到: {filepath}")
                return data
            else:
                logging.warning(f"未获取到 {symbol} 的{period}分钟数据")
                return None
                
        except Exception as e:
            logging.error(f"获取{period}分钟数据失败: {e}")
            return None
    
    def batch_fetch_all_contracts(self, period: str = "60", 
                                 force_update: bool = False,
                                 max_contracts: int = None) -> Dict[str, Optional[pd.DataFrame]]:
        """批量获取所有合约数据"""
        contracts = self.get_all_contracts()
        if contracts is None:
            return {}
        
        results = {}
        count = 0
        
        for _, contract in contracts.iterrows():
            symbol = contract['symbol']
            exchange = contract['exchange']
            name = contract['name']
            
            logging.info(f"处理合约 [{count+1}/{len(contracts)}]: {symbol} ({name}) - {self.exchange_mapping.get(exchange, exchange)}")
            
            data = self.get_contract_minute_data(symbol, period, force_update)
            results[symbol] = data
            
            count += 1
            if max_contracts and count >= max_contracts:
                logging.info(f"已达到最大合约数量限制: {max_contracts}")
                break
            
            # 添加短暂延迟避免请求过于频繁
            time.sleep(1)
        
        return results
    
    def get_market_summary(self, period: str = "60") -> Dict:
        """获取市场数据摘要"""
        summary = {
            'total_contracts': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'exchange_stats': {},
            'data_points': 0
        }
        
        contracts = self.get_all_contracts()
        if contracts is None:
            return summary
        
        summary['total_contracts'] = len(contracts)
        
        for _, contract in contracts.iterrows():
            symbol = contract['symbol']
            exchange = contract['exchange']
            
            filepath = self.data_dir / f"minute_{symbol}_{period}.csv"
            if filepath.exists():
                try:
                    data = pd.read_csv(filepath)
                    summary['successful_fetches'] += 1
                    summary['data_points'] += len(data)
                    
                    if exchange not in summary['exchange_stats']:
                        summary['exchange_stats'][exchange] = {
                            'count': 0,
                            'data_points': 0
                        }
                    summary['exchange_stats'][exchange]['count'] += 1
                    summary['exchange_stats'][exchange]['data_points'] += len(data)
                    
                except Exception as e:
                    summary['failed_fetches'] += 1
            else:
                summary['failed_fetches'] += 1
        
        return summary


def main():
    """主函数"""
    if not HAS_AKSHARE:
        print("请先安装AkShare: pip install akshare")
        return
    
    parser = argparse.ArgumentParser(description='AkShare批量数据获取工具')
    
    parser.add_argument('--period', '-p', default='60', choices=['1', '5', '15', '30', '60'], 
                       help='数据周期 (分钟)')
    parser.add_argument('--data-dir', '-d', default='data/akshare', 
                       help='数据存储目录')
    parser.add_argument('--force', '-f', action='store_true', 
                       help='强制更新数据')
    parser.add_argument('--max-contracts', '-m', type=int, default=None,
                       help='最大合约数量（用于测试）')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='显示数据摘要')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        fetcher = AkshareBatchFetcher(data_dir=args.data_dir)
    except ImportError as e:
        print(e)
        return
    
    if args.summary:
        print("\n=== 市场数据摘要 ===")
        summary = fetcher.get_market_summary(args.period)
        print(f"总合约数量: {summary['total_contracts']}")
        print(f"成功获取数据: {summary['successful_fetches']}")
        print(f"失败获取数据: {summary['failed_fetches']}")
        print(f"总数据点数: {summary['data_points']}")
        
        print("\n交易所统计:")
        for exchange, stats in summary['exchange_stats'].items():
            chinese_name = fetcher.exchange_mapping.get(exchange, exchange)
            print(f"  {exchange} ({chinese_name}): {stats['count']}个合约, {stats['data_points']}条数据")
    
    else:
        print(f"\n=== 开始批量获取所有大宗商品数据 ===")
        print(f"数据周期: {args.period}分钟")
        print(f"数据目录: {args.data_dir}")
        if args.max_contracts:
            print(f"最大合约数量: {args.max_contracts}")
        
        results = fetcher.batch_fetch_all_contracts(
            period=args.period,
            force_update=args.force,
            max_contracts=args.max_contracts
        )
        
        successful = sum(1 for data in results.values() if data is not None)
        total = len(results)
        
        print(f"\n=== 批量获取完成 ===")
        print(f"处理合约总数: {total}")
        print(f"成功获取数据: {successful}")
        print(f"失败数量: {total - successful}")
        print(f"数据已保存到: {args.data_dir}")


if __name__ == "__main__":
    main()