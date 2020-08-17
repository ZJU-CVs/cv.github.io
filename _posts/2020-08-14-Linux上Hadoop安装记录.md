---
layout:     post
title:      Linux上Hadoop安装记录
subtitle:   
date:       2020-08-14
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - 架构

---



#### 安装

安装包下载仓库：http://mirrors.hust.edu.cn/apache/hadoop/core/stable/

```bash
wget http://mirrors.hust.edu.cn/apache/hadoop/core/stable/hadoop-3.2.1.tar.gz
tar -zxvf hadoop-3.2.1.tar.gz
```



#### 配置

在`hadoop-3.2.1/etc/hadoop/`目录下，修改配置文件：hadoop-env.sh，core-site.xml，mapred-site.xml.template，hdfs-site.xml

##### core-site.xml 配置 

其中的hadoop.tmp.dir的路径可以根据自己的习惯进行设置

```xml
<configuration>
  <property>
    <name>hadoop.tmp.dir</name>
    <value>file:/home/hadoop/hadoop/tmp</value>  
    <description>Abase for other temporary directories.</description>
  </property>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>
```



##### mapred-site.xml.template配置 

```xml
<configuration>
  <property>
    <name>mapred.job.tracker</name>
    <value>localhost:9001</value>
  </property>
</configuration>
```



##### hdfs-site.xml配置 

其中dfs.namenode.name.dir和dfs.datanode.data.dir的路径可以自由设置，最好在hadoop.tmp.dir的目录下面。

```xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>file:/home/hadoop/hadoop/tmp/dfs/name</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file:/home/hadoop/hadoop/tmp/dfs/data</value>
  </property>
</configuration>
```



##### 常见问题



export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64







#### 执行



```
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
PATH=$PATH:$JAVA_HOME/bin
export HADOOP_INSTALL=/data0/JY/arch/hadoop-3.2.1
PATH=$PATH:$HADOOP_INSTALL/bin
PATH=$PATH:$HADOOP_INSTALL/sbin
export PATH
```



##### 常见问题

sudo apt-get install openjdk-8-jdk





#### 可视化工具

 Zeppelin和Hue