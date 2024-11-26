@echo off
set REPO_ROOT=%cd%
echo work space: %REPO_ROOT%

REM 创建目录并下载 SuperPoint 的模型
mkdir "%REPO_ROOT%\data\models\extractors\SuperPoint"
cd "%REPO_ROOT%\data\models\extractors\SuperPoint"
curl -O https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth

REM 创建目录并下载 SuperGlue 的模型
mkdir "%REPO_ROOT%\data\models\matchers\SuperGlue"
cd "%REPO_ROOT%\data\models\matchers\SuperGlue"
curl -O https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth

echo Download completed!
pause
