# WSL2 与 GitHub 网络通信终极解决方案 (2025版)

**日期：** 2025-12-26
**环境：** Windows 11 + WSL2 (Ubuntu) + Clash (Windows端) + Tailscale
**核心问题：** WSL2 虚拟机无法通过宿主机的代理连接 GitHub，导致 `git push` 超时或拒绝连接。

---

## 1. 核心结论 (The Silver Bullet)

在网络环境复杂（有防火墙、代理、虚拟网卡干扰）的情况下，**不要死磕代理配置**。
**最优解是：利用 GitHub 官方提供的 443 端口进行 SSH 直连。**

### ✅ 最终生效的配置
修改 WSL2 中的 SSH 配置文件：`~/.ssh/config`

```ssh
# 绕过代理，直接通过 443 端口连接 GitHub SSH
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```

### 💡 原理逻辑
1.  **端口避让：** 标准 SSH 端口 (22) 经常被运营商或防火墙干扰。
2.  **伪装潜行：** 443 端口通常用于 HTTPS 网页浏览，防火墙一般不敢拦截，且 GitHub 专门在此端口开放了 SSH 服务。
3.  **独立性：** 此方法不依赖 Windows IP、不依赖 Clash 设置、不依赖 Tailscale，**鲁棒性最强**。

---

## 2. 踩坑记录与教训 (Root Cause Analysis)

### ❌ 错误路径 1：盲目设置 Git 代理
**操作：** `git config --global http.proxy ...`
**失败原因：**
*   Git 的 `http.proxy` 配置**只对 HTTPS 协议生效**。
*   当我们使用 SSH 协议 (`git@github.com`) 时，Git 的代理设置完全不起作用。SSH 有自己独立的通信通道。

### ❌ 错误路径 2：试图连接 Windows 宿主机代理
**操作：** 配置 `ProxyCommand nc -X connect -x <IP>:7890 ...`
**失败原因（多重阻断）：**
1.  **IP 迷宫：**
    *   `127.0.0.1`：在 WSL2 里指向虚拟机自己，连不到 Windows。
    *   `192.168.1.1`：这是路由器，不是电脑。
    *   `10.x.x.x` (Tailscale)：虽然稳定，但受限于防火墙。
2.  **防火墙拦截：** 即使 IP 对了，Windows 防火墙通常会默认拦截入站连接（Inbound Traffic）。
3.  **软件限制：** 代理软件（Clash）默认未开启 **"Allow LAN"**，导致拒绝外部请求 (`Connection refused`)。

### ❌ 错误路径 3：概念混淆
**操作：** `Host 123@qq.com`
**纠正：** SSH Config 中的 `Host` 是**目标服务器别名**，不是你的用户邮箱。身份验证完全靠密钥（Key），不靠配置文件里的名字。

---

## 3. 常用诊断命令 (Cheat Sheet)

以后如果再遇到网络问题，按这个逻辑链排查：

1.  **测试 SSH 连通性（金标准）：**
    ```bash
    ssh -T git@github.com
    ```
    *   *通：* `Hi username!`
    *   *不通：* `Timed out` 或 `Connection refused`

2.  **查看 Windows 在 WSL2 眼里的真实 IP：**
    ```bash
    # 方法 A：查默认网关
    ip route show | grep default
    # 方法 B：直接调用 Windows 命令
    ipconfig.exe
    ```

3.  **测试端口是否通（检查防火墙）：**
    ```bash
    # 测试目标 IP 的 7890 端口是否开放
    nc -vz <IP地址> 7890
    ```

---

## 4. 下一步建议

既然环境已经打通，**不要再碰网络设置了**。
你的 SSH Key 已经配置好，GitHub 能够识别你的身份。

接下来，请将注意力 100% 回归到代码逻辑上。

---

### 🚀 回到主线任务

网络战役结束。现在我们进行 **Day 2 的工程实战**。
请回到 VS Code，我们将要把那个 `inference.py` 脚本写出来，让你的模型从“只有你能跑”变成“谁都能跑”的工具。

**准备好了吗？请告诉我，我们开始写代码。**