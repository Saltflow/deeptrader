# coding=utf-8
# 策略基类模块

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: PositionSide
    quantity: int
    entry_price: float
    entry_time: pd.Timestamp
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_price(self, price: float):
        """更新当前价格并计算未实现盈亏"""
        self.current_price = price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
        else:
            self.unrealized_pnl = 0.0


@dataclass
class Order:
    """订单信息"""
    symbol: str
    side: PositionSide
    quantity: int
    order_type: str  # 'market', 'limit'
    price: Optional[float] = None
    status: str = 'pending'  # 'pending', 'filled', 'cancelled'
    order_id: str = ""
    timestamp: pd.Timestamp = None


class Strategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.cash: float = 1000000.0  # 初始资金
        self.equity: float = self.cash
        self.commission_rate: float = 0.0003  # 手续费率
        self.slippage: float = 0.0001  # 滑点
        
        # 性能指标
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [self.equity]
        self.timestamps: List[pd.Timestamp] = []
        
        logger.info(f"初始化策略: {self.name}")
    
    @abstractmethod
    def on_bar(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp):
        """
        每个bar的回调函数
        
        Args:
            data: 所有合约的数据字典 {symbol: DataFrame}
            timestamp: 当前时间戳
        """
        pass
    
    def on_order_filled(self, order: Order, fill_price: float, fill_time: pd.Timestamp):
        """订单成交回调"""
        # 计算成交金额和手续费
        trade_value = order.quantity * fill_price
        commission = trade_value * self.commission_rate
        
        # 更新资金
        if order.side == PositionSide.LONG:
            self.cash -= trade_value + commission
        elif order.side == PositionSide.SHORT:
            self.cash += trade_value - commission
        
        # 更新持仓
        self._update_position(order, fill_price, fill_time)
        
        # 记录交易
        trade_record = {
            'timestamp': fill_time,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': fill_price,
            'commission': commission,
            'trade_value': trade_value
        }
        self.trades.append(trade_record)
        
        logger.info(f"订单成交: {order.symbol} {order.side.value} {order.quantity} @ {fill_price}")
    
    def _update_position(self, order: Order, fill_price: float, fill_time: pd.Timestamp):
        """更新持仓"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            # 新建持仓
            self.positions[symbol] = Position(
                symbol=symbol,
                side=order.side,
                quantity=order.quantity,
                entry_price=fill_price,
                entry_time=fill_time
            )
        else:
            # 更新现有持仓
            current_pos = self.positions[symbol]
            
            if current_pos.side == order.side:
                # 同向加仓
                total_quantity = current_pos.quantity + order.quantity
                avg_price = (current_pos.entry_price * current_pos.quantity + 
                           fill_price * order.quantity) / total_quantity
                current_pos.quantity = total_quantity
                current_pos.entry_price = avg_price
            else:
                # 反向平仓或反向开仓
                if order.quantity >= current_pos.quantity:
                    # 完全平仓
                    realized_pnl = self._calculate_pnl(current_pos, fill_price, order.quantity)
                    current_pos.realized_pnl += realized_pnl
                    
                    if order.quantity > current_pos.quantity:
                        # 反向开仓
                        remaining_quantity = order.quantity - current_pos.quantity
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            side=order.side,
                            quantity=remaining_quantity,
                            entry_price=fill_price,
                            entry_time=fill_time
                        )
                    else:
                        # 完全平仓
                        del self.positions[symbol]
                else:
                    # 部分平仓
                    realized_pnl = self._calculate_pnl(current_pos, fill_price, order.quantity)
                    current_pos.realized_pnl += realized_pnl
                    current_pos.quantity -= order.quantity
    
    def _calculate_pnl(self, position: Position, exit_price: float, quantity: int) -> float:
        """计算盈亏"""
        if position.side == PositionSide.LONG:
            return (exit_price - position.entry_price) * quantity
        elif position.side == PositionSide.SHORT:
            return (position.entry_price - exit_price) * quantity
        return 0.0
    
    def place_order(self, symbol: str, side: PositionSide, quantity: int, 
                   order_type: str = 'market', price: Optional[float] = None) -> Order:
        """下达订单"""
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            order_id=f"order_{len(self.orders) + 1}",
            timestamp=pd.Timestamp.now()
        )
        self.orders.append(order)
        
        logger.info(f"下达订单: {symbol} {side.value} {quantity} {order_type}")
        return order
    
    def update_equity(self, current_prices: Dict[str, float], timestamp: pd.Timestamp):
        """更新权益曲线"""
        # 更新持仓市值
        position_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_price(current_prices[symbol])
                position_value += position.current_price * position.quantity
                
                # 根据持仓方向调整市值计算
                if position.side == PositionSide.SHORT:
                    position_value = -position_value
        
        # 计算总权益
        self.equity = self.cash + position_value
        self.equity_curve.append(self.equity)
        self.timestamps.append(timestamp)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if len(self.equity_curve) < 2:
            return {}
        
        # 计算收益率
        returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        
        # 基本指标
        total_return = (self.equity_curve[-1] / self.equity_curve[0] - 1) * 100
        annual_return = total_return / (len(self.equity_curve) / 252)  # 假设252个交易日
        
        # 风险指标
        volatility = np.std(returns) * np.sqrt(252) * 100  # 年化波动率
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # 索提诺比率
        sortino_ratio = self._calculate_sortino_ratio(returns, annual_return)
        
        # 最大回撤
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak * 100
        max_drawdown = np.min(drawdown)
        
        return {
            'initial_capital': self.equity_curve[0],
            'final_equity': self.equity_curve[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor()
        }
    
    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trades 
                           if (trade['side'] == 'long' and trade['price'] > 0) or 
                              (trade['side'] == 'short' and trade['price'] < 0))
        return winning_trades / len(self.trades) * 100
    
    def _calculate_profit_factor(self) -> float:
        """计算盈亏比"""
        if not self.trades:
            return 0.0
        
        gross_profit = sum(trade['trade_value'] for trade in self.trades 
                         if (trade['side'] == 'long' and trade['price'] > 0) or 
                            (trade['side'] == 'short' and trade['price'] < 0))
        gross_loss = abs(sum(trade['trade_value'] for trade in self.trades 
                           if (trade['side'] == 'long' and trade['price'] < 0) or 
                              (trade['side'] == 'short' and trade['price'] > 0)))
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, annual_return: float) -> float:
        """计算索提诺比率
        
        Args:
            returns: 日收益率数组
            annual_return: 年化收益率
            
        Returns:
            索提诺比率
        """
        if len(returns) == 0:
            return 0.0
        
        # 计算下行偏差（只考虑负收益）
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            # 如果没有下行风险，索提诺比率为无穷大
            return float('inf')
        
        # 计算年化下行偏差
        downside_deviation = np.std(downside_returns) * np.sqrt(252) * 100
        
        # 计算索提诺比率
        if downside_deviation != 0:
            sortino_ratio = annual_return / downside_deviation
        else:
            sortino_ratio = float('inf')
        
        return sortino_ratio
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """获取持仓摘要"""
        summary = {}
        for symbol, position in self.positions.items():
            summary[symbol] = {
                'side': position.side.value,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl
            }
        return summary


class MovingAverageCrossover(Strategy):
    """移动平均线交叉策略示例"""
    
    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        super().__init__(name=f"MA_Crossover_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.data_history: Dict[str, pd.DataFrame] = {}
    
    def on_bar(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp):
        """移动平均线交叉策略逻辑"""
        for symbol, df in data.items():
            if symbol not in self.data_history:
                self.data_history[symbol] = df.copy()
            else:
                self.data_history[symbol] = pd.concat([
                    self.data_history[symbol], df
                ]).drop_duplicates().sort_index()
            
            # 确保有足够的数据
            if len(self.data_history[symbol]) < self.slow_period:
                continue
            
            # 计算移动平均线
            close_prices = self.data_history[symbol]['close']
            fast_ma = close_prices.rolling(window=self.fast_period).mean().iloc[-1]
            slow_ma = close_prices.rolling(window=self.slow_period).mean().iloc[-1]
            
            current_price = close_prices.iloc[-1]
            
            # 交易信号
            if fast_ma > slow_ma:
                # 金叉信号，做多
                if symbol not in self.positions or self.positions[symbol].side != PositionSide.LONG:
                    # 平掉空头仓位（如果有）
                    if symbol in self.positions and self.positions[symbol].side == PositionSide.SHORT:
                        self.place_order(symbol, PositionSide.LONG, self.positions[symbol].quantity)
                    # 开多头仓位
                    quantity = int(self.equity * 0.1 / current_price)  # 10%仓位
                    if quantity > 0:
                        self.place_order(symbol, PositionSide.LONG, quantity)
            
            elif fast_ma < slow_ma:
                # 死叉信号，做空
                if symbol not in self.positions or self.positions[symbol].side != PositionSide.SHORT:
                    # 平掉多头仓位（如果有）
                    if symbol in self.positions and self.positions[symbol].side == PositionSide.LONG:
                        self.place_order(symbol, PositionSide.SHORT, self.positions[symbol].quantity)
                    # 开空头仓位
                    quantity = int(self.equity * 0.1 / current_price)  # 10%仓位
                    if quantity > 0:
                        self.place_order(symbol, PositionSide.SHORT, quantity)
            
            # 更新当前价格用于权益计算
            current_prices = {symbol: current_price}
            self.update_equity(current_prices, timestamp)