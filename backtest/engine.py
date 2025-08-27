# coding=utf-8
# 回测引擎核心模块

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .data_loader import BacktestDataLoader
from .strategy import Strategy, Order, PositionSide

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str
    end_date: str
    initial_capital: float = 1000000.0
    commission_rate: float = 0.0003
    slippage: float = 0.0001
    data_frequency: str = "60min"  # 数据频率
    benchmark: Optional[str] = None  # 基准合约


class BacktestEngine:
    """回测引擎核心类"""
    
    def __init__(self, config: BacktestConfig, data_dir: str = "data/akshare"):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
            data_dir: 数据目录
        """
        self.config = config
        self.data_loader = BacktestDataLoader(data_dir)
        
        # 回测状态
        self.current_time: Optional[pd.Timestamp] = None
        self.current_bar_data: Dict[str, pd.Series] = {}
        self.strategies: List[Strategy] = []
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        
        # 数据存储
        self.all_data: Dict[str, pd.DataFrame] = {}
        self.time_index: pd.DatetimeIndex = None
        self.current_index: int = 0
        
        # 性能记录
        self.equity_records: List[Dict] = []
        self.trade_records: List[Dict] = []
        
        logger.info(f"初始化回测引擎，时间范围: {config.start_date} - {config.end_date}")
    
    def add_strategy(self, strategy: Strategy):
        """添加策略"""
        # 设置策略初始参数
        strategy.cash = self.config.initial_capital
        strategy.equity = self.config.initial_capital
        strategy.commission_rate = self.config.commission_rate
        strategy.slippage = self.config.slippage
        
        self.strategies.append(strategy)
        logger.info(f"添加策略: {strategy.name}")
    
    def add_data(self, symbol: str, data: Optional[pd.DataFrame] = None):
        """
        添加数据
        
        Args:
            symbol: 合约代码
            data: 数据DataFrame，如果为None则自动加载
        """
        if data is None:
            # 解析频率参数
            period = "60"  # 默认60分钟
            if "min" in self.config.data_frequency:
                period = self.config.data_frequency.replace("min", "")
            
            data = self.data_loader.load_minute_data(
                symbol=symbol,
                period=period
            )
        
        if data is not None and len(data) > 0:
            self.all_data[symbol] = data
            logger.info(f"添加数据: {symbol}, {len(data)} 条记录")
        else:
            logger.warning(f"未能加载数据: {symbol}")
    
    def prepare_data(self):
        """准备数据，同步时间轴"""
        if not self.all_data:
            raise ValueError("没有可用的数据，请先添加数据")
        
        # 获取所有时间戳并排序
        all_timestamps = set()
        for symbol, data in self.all_data.items():
            all_timestamps.update(data.index)
        
        self.time_index = pd.DatetimeIndex(sorted(all_timestamps))
        logger.info(f"准备数据完成，时间范围: {self.time_index[0]} - {self.time_index[-1]}")
        logger.info(f"总时间点数量: {len(self.time_index)}")
    
    def get_current_data(self) -> Dict[str, pd.DataFrame]:
        """获取当前时间点的数据"""
        current_data = {}
        
        for symbol, data in self.all_data.items():
            # 获取当前时间点及之前的数据
            mask = data.index <= self.current_time
            if mask.any():
                current_data[symbol] = data[mask]
        
        return current_data
    
    def get_current_prices(self) -> Dict[str, float]:
        """获取当前价格"""
        current_prices = {}
        
        for symbol, data in self.all_data.items():
            # 获取最新的价格
            mask = data.index <= self.current_time
            if mask.any():
                latest_data = data[mask].iloc[-1]
                current_prices[symbol] = latest_data['close']
        
        return current_prices
    
    def process_orders(self):
        """处理订单"""
        current_prices = self.get_current_prices()
        filled_orders = []
        
        for order in self.pending_orders:
            if order.symbol in current_prices:
                # 简单的市价单成交逻辑
                fill_price = current_prices[order.symbol]
                
                # 应用滑点
                if order.side == PositionSide.LONG:
                    fill_price *= (1 + self.config.slippage)
                elif order.side == PositionSide.SHORT:
                    fill_price *= (1 - self.config.slippage)
                
                # 标记订单为已成交
                order.status = 'filled'
                filled_orders.append(order)
                
                # 通知策略订单成交
                for strategy in self.strategies:
                    if order in strategy.orders:
                        strategy.on_order_filled(order, fill_price, self.current_time)
                
                # 记录交易
                trade_record = {
                    'timestamp': self.current_time,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'price': fill_price,
                    'commission': fill_price * order.quantity * self.config.commission_rate,
                    'strategy': getattr(order, 'strategy_name', 'unknown')
                }
                self.trade_records.append(trade_record)
        
        # 移除已成交的订单
        for order in filled_orders:
            if order in self.pending_orders:
                self.pending_orders.remove(order)
            self.filled_orders.append(order)
    
    def update_portfolio(self):
        """更新投资组合"""
        current_prices = self.get_current_prices()
        
        for strategy in self.strategies:
            # 更新策略权益
            strategy.update_equity(current_prices, self.current_time)
            
            # 记录权益曲线
            equity_record = {
                'timestamp': self.current_time,
                'strategy': strategy.name,
                'cash': strategy.cash,
                'equity': strategy.equity,
                'positions': len(strategy.positions)
            }
            self.equity_records.append(equity_record)
    
    def run_backtest(self):
        """运行回测"""
        if self.time_index is None or len(self.time_index) == 0:
            logger.error("数据未准备好，请先调用 prepare_data()")
            return
        
        if not self.strategies:
            logger.error("没有策略，请先添加策略")
            return
        
        logger.info("开始回测...")
        total_bars = len(self.time_index)
        
        for i, timestamp in enumerate(self.time_index):
            self.current_time = timestamp
            self.current_index = i
            
            # 获取当前数据
            current_data = self.get_current_data()
            
            # 执行策略逻辑
            for strategy in self.strategies:
                try:
                    # 收集策略的新订单
                    old_order_count = len(strategy.orders)
                    strategy.on_bar(current_data, timestamp)
                    
                    # 添加新订单到待处理列表
                    new_orders = strategy.orders[old_order_count:]
                    for order in new_orders:
                        order.strategy_name = strategy.name
                        self.pending_orders.append(order)
                        
                except Exception as e:
                    logger.error(f"策略 {strategy.name} 执行错误: {e}")
            
            # 处理订单
            self.process_orders()
            
            # 更新投资组合
            self.update_portfolio()
            
            # 进度显示
            if i % 100 == 0 or i == total_bars - 1:
                progress = (i + 1) / total_bars * 100
                logger.info(f"回测进度: {progress:.1f}% ({i+1}/{total_bars})")
        
        logger.info("回测完成")
        self._generate_report()
    
    def _generate_report(self):
        """生成回测报告"""
        logger.info("=" * 50)
        logger.info("回测报告")
        logger.info("=" * 50)
        
        for strategy in self.strategies:
            logger.info(f"\n策略: {strategy.name}")
            
            # 基本信息
            logger.info(f"初始资金: {self.config.initial_capital:,.2f}")
            logger.info(f"最终权益: {strategy.equity:,.2f}")
            logger.info(f"总收益: {strategy.equity - self.config.initial_capital:,.2f}")
            logger.info(f"收益率: {(strategy.equity / self.config.initial_capital - 1) * 100:.2f}%")
            
            # 交易统计
            strategy_trades = [t for t in self.trade_records if t.get('strategy') == strategy.name]
            logger.info(f"总交易次数: {len(strategy_trades)}")
            
            # 性能指标
            performance = strategy.get_performance_metrics()
            if performance:
                logger.info(f"年化收益率: {performance.get('annual_return', 0):.2f}%")
                logger.info(f"波动率: {performance.get('volatility', 0):.2f}%")
                logger.info(f"夏普比率: {performance.get('sharpe_ratio', 0):.2f}")
                logger.info(f"最大回撤: {performance.get('max_drawdown', 0):.2f}%")
                logger.info(f"胜率: {performance.get('win_rate', 0):.2f}%")
    
    def get_results(self) -> Dict[str, Any]:
        """获取回测结果"""
        results = {
            'config': self.config,
            'strategies': {},
            'equity_records': self.equity_records,
            'trade_records': self.trade_records,
            'start_time': self.time_index[0] if len(self.time_index) > 0 else None,
            'end_time': self.time_index[-1] if len(self.time_index) > 0 else None
        }
        
        for strategy in self.strategies:
            results['strategies'][strategy.name] = {
                'performance': strategy.get_performance_metrics(),
                'positions': strategy.get_positions_summary(),
                'final_equity': strategy.equity,
                'total_trades': len([t for t in self.trade_records if t.get('strategy') == strategy.name])
            }
        
        return results
    
    def save_results(self, output_path: str):
        """保存结果到文件"""
        results = self.get_results()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存权益曲线
        if self.equity_records:
            equity_df = pd.DataFrame(self.equity_records)
            equity_df.to_csv(output_path.parent / f"{output_path.stem}_equity.csv", index=False)
        
        # 保存交易记录
        if self.trade_records:
            trade_df = pd.DataFrame(self.trade_records)
            trade_df.to_csv(output_path.parent / f"{output_path.stem}_trades.csv", index=False)
        
        # 保存摘要报告
        summary = {
            'backtest_period': f"{self.config.start_date} to {self.config.end_date}",
            'initial_capital': self.config.initial_capital,
            'strategies': {}
        }
        
        for strategy_name, strategy_result in results['strategies'].items():
            summary['strategies'][strategy_name] = strategy_result
        
        import json
        with open(output_path.parent / f"{output_path.stem}_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"结果已保存到: {output_path.parent}")


def create_simple_backtest(symbols: List[str], strategy_class, strategy_params: Dict = None,
                          start_date: str = "2024-01-01", end_date: str = "2024-12-31",
                          initial_capital: float = 1000000.0) -> BacktestEngine:
    """
    创建简单回测
    
    Args:
        symbols: 合约代码列表
        strategy_class: 策略类
        strategy_params: 策略参数
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
        
    Returns:
        配置好的回测引擎
    """
    # 创建配置
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    # 创建引擎
    engine = BacktestEngine(config)
    
    # 添加数据
    for symbol in symbols:
        engine.add_data(symbol)
    
    # 创建策略
    if strategy_params:
        strategy = strategy_class(**strategy_params)
    else:
        strategy = strategy_class()
    
    engine.add_strategy(strategy)
    
    # 准备数据
    engine.prepare_data()
    
    return engine