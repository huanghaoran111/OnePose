from pytorch_lightning import LightningModule, Callback, Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

import hydra
from omegaconf import DictConfig
from typing import List
from src.utils import template_utils as utils

import warnings
warnings.filterwarnings('ignore')


def train(config: DictConfig):
    # 这段代码实现了一个基于 PyTorch Lightning 的模型训练流程，使用了 Hydra 配置管理工具来动态加载配置。

    # 配置打印和随机种子设置
    if config['print_config']:
        utils.print_config(config)
    # 打印配置，用于调试和确认参数
    if "seed" in config:
        seed_everything(config['seed'])
    # 设置随机种子，确保训练的结果可复现

    # Init PyTorch Lightning model ⚡
    model: LightningModule = hydra.utils.instantiate(config['model'])
    # 使用 Hydra 配置系统实例化 PyTorch Lightning 的 LightningModule
    # config['model'] 包含模型的定义和初始化参数
    
    # Init PyTorch Lightning datamodule ⚡
    datamodule: LightningModule = hydra.utils.instantiate(config['datamodule'])
    # 实例化数据模块 LightningDataModule
    datamodule.setup()
    # 调用 setup() 方法，完成数据加载和预处理

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config['callbacks'].items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))
    # 从配置中加载所有回调，并实例化
    # 回调用于扩展训练逻辑，如学习率调度、模型保存、早停等
    
    # Init PyTorch Lightning loggers ⚡
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config['logger'].items():
            if "_target_" in lg_conf:
                logger.append(hydra.utils.instantiate(lg_conf))
    # 从配置中加载日志记录器，用于记录训练过程中的指标、模型参数等
    # 支持多种日志记录工具（如 TensorBoard、WandB 等）

    # Init PyTorch Lightning trainer ⚡
    trainer: Trainer = hydra.utils.instantiate(
        config['trainer'], callbacks=callbacks, logger=logger
    )
    # 使用 Hydra 实例化 PyTorch Lightning 的 Trainer
    # 训练器负责整个训练过程的管理和调度

    # Send some parameters from config to all lightning loggers 
    utils.log_hparams_to_all_loggers(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger
    )
    # 将超参数信息（如模型、数据和训练器的配置）记录到所有日志记录器中，便于可视化和分析

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)
    # 使用 Trainer.fit 开始训练
    # model 和 datamodule 提供了训练所需的逻辑和数据

    # Evaluate model on test set after training
    # trainer.test()
    
    # Make sure everything closed properly 
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger
    )
    # 结束训练后的一些清理操作，如释放资源、保存模型等

    # Return best achieved metric score for optuna
    optimized_metric = config.get("optimized_metric", None)
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
    # 返回指定的优化指标值，便于与超参数调优工具（如 Optuna）集成
    # 如果配置中包含 optimized_metric，从训练回调结果中提取对应的指标值
        

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    return train(config)


if __name__ == "__main__":
    main()