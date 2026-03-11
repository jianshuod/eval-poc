#!/bin/bash
# 一键测评环境设置脚本
# 自动生成，请根据实际情况修改

set -e

# === xstest: 申请 HuggingFace 数据集访问权限 ===
# 1. 访问上述链接，点击 'Access repository' 申请访问
# 2. 等待审批通过（通常即时批准）
# 3. 设置环境变量: export HF_TOKEN=<your_token>
#    获取 Token: https://huggingface.co/settings/tokens

# === agentdojo: 启动/连接 Kubernetes 集群 ===
# 确保 Kubernetes 集群可访问。agentdojo 使用 K8s 沙箱执行代理安全测试。
# 首次使用可能需要:
#   1. 安装 kubectl: https://kubernetes.io/docs/tasks/tools/
#   2. 配置 kubeconfig 文件 (~/.kube/config)
#   3. 验证连接: kubectl cluster-info
kubectl cluster-info

# === gaia: 申请 HuggingFace GAIA 数据集访问权限 ===
# 1. 访问上述链接，点击 'Access repository' 申请访问
# 2. 等待审批通过（通常即时批准）
# 3. 设置环境变量: export HF_TOKEN=<your_token>
#    获取 Token: https://huggingface.co/settings/tokens

echo "设置完成！"