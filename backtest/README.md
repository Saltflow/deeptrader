# 回测系统使用指南

## 概述

本回测系统是一个基于Python的量化交易回测框架，支持多策略、多合约的回测分析。

## 快速开始

### 1. 运行示例回测
```bash
python backtest/example.py
```

### 2. 创建简单回测
```python
from backtest.engine import create_simple_backtest
from backtest.strategy import MovingAverageCrossover

# 创建移动平均线交叉策略回测
engine = create_simple_backtest(
    symbols=['RB0'],           # 螺纹钢主力合约
    strategy_class=MovingAverageCrossover,
    strategy_params={'fast_period': 5, 'slow_period': 20},
    start_date="2024-12-01",
    end_date="2025-08-26",
    initial_capital=1000000.0
)

# 运行回测
engine.run_backtest()

# 保存结果
engine.save_results("results/my_backtest")
```

### 3. 自定义回测配置
```python
from backtest.engine import BacktestEngine, BacktestConfig
from backtest.strategy import MovingAverageCrossover

# 创建回测配置
config = BacktestConfig(
    start_date="2024-12-01",
    end_date="2025-08-26",
    initial_capital=1000000.0,
    commission_rate=0.0003,    # 手续费率 0.03%
    slippage=0.0001,           # 滑点 0.01%
    data_frequency="60min"     # 数据频率
)

# 创建回测引擎
engine = BacktestEngine(config)

# 添加数据
engine.add_data('RB0')        # 螺纹钢主力合约
engine.add_data('rb2410')     # 螺纹钢2410合约

# 添加策略
strategies = [
    MovingAverageCrossover(fast_period=5, slow_period=10),
    MovingAverageCrossover(fast_period=10, slow_period=20),
    MovingAverageCrossover(fast_period=5, slow_period=20),
]

for strategy in strategies:
    engine.add_strategy(strategy)

# 准备数据并运行回测
engine.prepare_data()
engine.run_backtest()

# 获取结果
results = engine.get_results()
```

## 数据结构

### 数据文件格式
回测数据存储在 `data/akshare/` 目录下，格式为：
- `minute_{symbol}_{period}.csv`

示例文件：`minute_RB0_60.csv`

### 数据列说明
- `datetime`: 时间戳
- `open`: 开盘价
- `high`: 最高价  
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量
- `hold`: 持仓量

## 可用合约

系统支持以下主力合约代码：
- `RB0` - 螺纹钢主力
- `I0` - 铁矿石主力  
- `HC0` - 热卷主力
- `CU0` - 铜主力
- `AL0` - 铝主力
- `ZN0` - 锌主力
- 以及其他所有国内期货品种主力合约

## 策略开发

### 创建自定义策略
```python
from backtest.strategy import Strategy, PositionSide
from backtest.order import Order

class MyCustomStrategy(Strategy):
    def __init__(self, param1=10, param2=20):
        super().__init__(name="MyCustomStrategy")
        self.param1 = param1
        self.param2 = param2
        
    def on_bar(self, data, timestamp):
        """每个bar的回调函数"""
        for symbol, df in data.items():
            if len(df) >= self.param2:
                # 计算指标
                close_prices = df['close']
                ma_fast = close_prices.rolling(self.param1).mean().iloc[-1]
                ma_slow = close_prices.rolling(self.param2).mean().iloc[-1]
                
                current_price = close_prices.iloc[-1]
                
                # 交易逻辑
                if ma_fast > ma_slow:
                    # 做多信号
                    quantity = int(self.equity * 0.1 / current_price)
                    if quantity > 0:
                        self.place_order(symbol, PositionSide.LONG, quantity)
                elif ma_fast < ma_slow:
                    # 做空信号
                    quantity = int(self.equity * 0.1 / current_price)
                    if quantity > 0:
                        self.place_order(symbol, PositionSide.SHORT, quantity)
```

## 性能指标

回测结果包含以下性能指标：
- `total_return`: 总收益率
- `annual_return`: 年化收益率
- `volatility`: 波动率
- `sharpe_ratio`: 夏普比率
- `max_drawdown`: 最大回撤
- `win_rate`: 胜率
- `profit_factor`: 盈亏比
- `total_trades`: 总交易次数

## 输出文件

回测完成后生成以下文件：
- `{name}_equity.csv`: 权益曲线数据
- `{name}_trades.csv`: 交易记录数据  
- `{name}_summary.json`: 回测摘要报告

## 注意事项

1. **数据质量**: 确保数据文件完整，无缺失值
2. **参数优化**: 建议进行参数敏感性分析
3. **过拟合风险**: 避免在单一品种上过度优化参数
4. **手续费设置**: 根据实际交易成本调整手续费率
5. **滑点设置**: 根据品种流动性调整滑点参数

## 故障排除

### 常见问题
1. **数据加载失败**: 检查数据文件是否存在
2. **内存不足**: 减少回测数据量或合约数量
3. **策略错误**: 检查策略逻辑是否正确

### 调试模式
启用详细日志输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```