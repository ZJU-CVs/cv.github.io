---
layout:     post
title:      玉泉Ubuntu连接有线网络教程
subtitle:   
date:       2020-09-02
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Tools
---



# 1. 静态IP设置

### 查询本机有线网卡名称及MAC地址

命令行中输入：

```
ifconfig -a
```

第一行中最左边的如`enp5s0`是有线网卡名称，`HWaddr`后面为MAC地址。

### 申请IP

登录[浙大学生公寓服务网](http://service.chinasinew.com/zjuauth.aspx?syscode=zjuserv&redirect_url=http%3a%2f%2fservice.chinasinew.com%2flogin.ashx&sign=D41EC7A9AF38DDF5F5C7FAED27B08CB1)，按照提示输入MAC地址申请IP，注意申请成功后等待24小时方可生效。

### 打开配置文件1

命令行中输入：

```
sudo gedit /etc/network/interfaces # 使用gedit编辑器打开
```

### 修改并保存

```
# interfaces(5) file used by ifup(8) and ifdown(8)
auto lo
iface lo inet loopback
# 以下为追加内容
# the primary network interface
auto enp5s0 #这里以及下一行的"enp5s0"根据上面查询得到的网卡名称替换
iface enp5s0 inet static
address 10.110.91.178 # IP地址、子网掩码和网关这三个参数根据上面申请IP的网站替换
netmask 255.255.255.0
gateway 10.110.91.1
dns-nameserver 10.10.0.21# 浙大默认的DNS
```

### 刷新IP

命令行中输入：

```
sudo ip addr flush enp5s0 #注意替换网卡名称
sudo systemctl restart networking.service
```

### 重启系统

命令行中输入： `reboot`

## 修改配置文件2

修改`/etc/NetworkManager/NetworkManager.conf`如下：

```
[main]
plugins=ifupdown,keyfile,ofono
dns=dnsmasq

[ifupdown]

# managed=false

managed=true 
```

### 重启network manager

```
sudo service network-manager restart
```

### 内网测试

```
ping 10.5.1.7 #浙大VPN服务器地址
```

如果可以ping通，说明IP配置正确。



# 2. VPN客户端连接

### 软件准备

考虑到非虚拟机环境下，新安装的Ubuntu可能没有无线网卡驱动，无法连接网络。在断网的情况下需要先下载好相关软件并用U盘导入。 网盘链接如下，如失效请私信我或在下方评论。 [Ubuntu下的VPN软件以及依赖的iproute安装包](http://pan.baidu.com/s/1cni30a)，感谢cc98相关用户提供。

### 安装

```
sudo dpkg -i iproute_3.12.0-2_all.deb 
sudo dpkg -i xl2tpd_1.1.12-zju2_amd64.deb
```

### 配置

`sudo vpn-connect -c` 按照提示输入用户名和密码： `Username`为原用户名+“@a”，如"3140106666@a"；`Password`为原密码。

### 使用方式

连接：`sudo vpn-connect #必须要以root身份运行` 断开：`sudo vpn-connect -d #同上`