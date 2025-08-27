# coding=utf-8
# 回测框架使用示例

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from backtest.engine import BacktestEngine, BacktestConfig, create_simple_backtest
from backtest.strategy import MovingAverageCrossover
from backtest.data_loader import BacktestDataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_ma_crossover_backtest():
    """运行移动平均线交叉策略回测"""
    logger.info("开始移动平均线交叉策略回测")
    
    # 设置回测参数
    symbols = ['RB0']  # 螺纹钢主力合约
    start_date = "2024-12-01"
    end_date = "2025-08-26"
    initial_capital = 1000000.0
    
    # 创建策略参数
    strategy_params = {
        'fast_period': 5,
        'slow_period': 20
    }
    
    try:
        # 创建简单回测
        engine = create_simple_backtest(
            symbols=symbols,
            strategy_class=MovingAverageCrossover,
            strategy_params=strategy_params,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        # 运行回测
        engine.run_backtest()
        
        # 保存结果
        engine.save_results("results/ma_crossover_backtest")
        
        return engine
        
    except Exception as e:
        logger.error(f"回测执行失败: {e}")
        return None


def run_custom_backtest():
    """运行自定义回测"""
    logger.info("开始自定义回测")
    
    # 创建回测配置
    config = BacktestConfig(
        start_date="2024-12-01",
        end_date="2025-08-26",
        initial_capital=1000000.0,
        commission_rate=0.0003,
        slippage=0.0001,
        data_frequency="60min"
    )
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    
    # 添加数据
    symbols = ['RB0', 'rb2410']
    for symbol in symbols:
        engine.add_data(symbol)
    
    # 创建多个策略进行对比
    strategies = [
        MovingAverageCrossover(fast_period=5, slow_period=10),
        MovingAverageCrossover(fast_period=10, slow_period=20),
        MovingAverageCrossover(fast_period=5, slow_period=20),
    ]
    
    for strategy in strategies:
        engine.add_strategy(strategy)
    
    # 准备数据
    engine.prepare_data()
    
    # 运行回测
    engine.run_backtest()
    
    # 获取结果
    results = engine.get_results()
    
    # 显示对比结果
    logger.info("\n策略对比结果:")
    for strategy_name, result in results['strategies'].items():
        performance = result['performance']
        logger.info(f"{strategy_name}:")
        logger.info(f"  最终权益: {result['final_equity']:,.2f}")
        logger.info(f"  总收益率: {performance.get('total_return', 0):.2f}%")
        logger.info(f"  年化收益率: {performance.get('annual_return', 0):.2f}%")
        logger.info(f"  夏普比率: {performance.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  最大回撤: {performance.get('max_drawdown', 0):.2f}%")
        logger.info(f"  总交易次数: {result['total_trades']}")
    
    # 保存结果
    engine.save_results("results/multi_strategy_backtest")
    
    return engine


def analyze_data():
    """分析数据质量"""
    logger.info("开始数据质量分析")
    
    data_loader = BacktestDataLoader()
    
    symbols = ['RB0', 'rb2410']
    for symbol in symbols:
        logger.info(f"\n分析合约: {symbol}")
        
        data = data_loader.load_minute_data(
            symbol=symbol,
            period="60"
        )
        
        if data is not None:
            logger.info(f"数据量: {len(data)} 条")
            logger.info(f"时间范围: {data.index[0]} 到 {data.index[-1]}")
            logger.info(f"价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")
            logger.info(f"成交量范围: {data['volume'].min():,} - {data['volume'].max():,}")
            
            # 检查数据完整性
            missing_data = data.isnull().sum()
            logger.info(f"缺失数据: {missing_data.sum()} 个")
            
            # 价格统计
            price_stats = data['close'].describe()
            logger.info(f"价格统计:\n{price_stats}")
        else:
            logger.warning(f"无法加载 {symbol} 的数据")


if __name__ == "__main__":
    # 分析数据
    analyze_data()
    
    print("\n" + "="*60)
    
    # 运行简单回测
    engine1 = run_ma_crossover_backtest()
    
    print("\n" + "="*60)
    
    # 运行多策略对比回测
    engine2 = run_custom_backtest()
    
    logger.info("示例运行完成")