---
layout:       post
title:        "协程库项目实现2"
author:       "Sutdown"
header-style: text
catalog:      true
tags:
    - coroutine
    - C++
---

# 协程库项目实现2 - thread，协程类



## thread

主要有两个类，`Semaphore`和`Thread`

#### Semaphore

信号量，实现PV操作，主要用于线程同步

#### Thread

1. 系统自动创建主线程t_thread

2. 由thread类创建的线程。

   m_thread 通常是线程类内部的成员变量，用来存储底层的线程标识符

   t_thread 可能是外部管理线程生命周期的对象或容器，它可以是线程池、线程列表、智能指针等，帮助你在类外部管理多个线程的创建、执行、销毁等操作。

   

## 协程类

- 非对称模型
- 有栈协程，独立栈。

对于协程类，其中需要什么。协程首先需要随时切换和恢复，这里采用的是**glibc的ucontext组件**。

### ucontext_t

这个类中有成员：

```cpp
// 当前上下文结束后下一个激活的上下文对象的指针，只在当前上下文是由makecontext创建时有效
struct ucontext_t *uc_link;
// 当前上下文的信号屏蔽掩码
sigset_t uc_sigmask;
// 当前上下文使用的栈内存空间，只在当前上下文是由makecontext创建时有效
stack_t uc_stack;
// 平台相关的上下文具体内容，包含寄存器的值
mcontext_t uc_mcontext;
```

函数：

```cpp
// 获取当前上下文
int getcontext(ucontext_t *ucp);

// 恢复ucp指向的上下文
int setcontext(const ucontext_t *ucp);

// 修改当前上下文指针ucp，将其与func函数绑定
void makecontext(ucontext_t *ucp, void (*func)(), int argc, ...);

// 恢复ucp指向的上下文，同时将当前上下文存储到oucp中
int swapcontext(ucontext_t *oucp, const ucontext_t *ucp);
```



> 对于该协程类，有栈 or 无栈？对称 or 非对称？

- 对于对称和非对称的话，**对称协程更为灵活，非对称协程更为简单易实现**。协程中一般存在协程调度器和协程两种角色，对称协程中相当于每个协程都要充当调度器的角色，程序设计复杂，程序的控制流也会复杂难以管理。

  常见的`js中的async/await`，`go中的coroutine`都是非对称协程，是因为非对称协程的切换过程是单项的，更适合事件驱动，任务队列等调度模型；但是c语言中的`ucontext`属于对称协程的经典实现，`boost.context`为对称协程的现代实现，更适合需要多个协程频繁通信的场景。

- [有栈协程和无栈协程](https://mthli.xyz/stackful-stackless/)有栈和无栈的本质区别在于是否可以在任意嵌套函数中被挂起。一般有栈可以被挂起，无栈则不行。有栈比较适用于功能强大，支持嵌套调用和复杂控制流，灵活的操作上下文的需求，比如`boost.COntext`；无栈由于存储在内存中，适用于内存占用少，实现简单的场景，比如JavaScript `async`/`await` 和 `Promise`，Erlang 和 Go的`Goroutine`。

> 共享栈 or 独立栈？



这里我们的协程类，采用的是非对称模型，有栈协程。因此可以推导出所需要的私有成员：

```cpp
private:
    uint64_t m_id = 0;
    State m_state = READY;

    ucontext_t m_ctx;

    uint32_t m_stacksze = 0;    // 栈大小
    void *m_stack = nullptr;    // 栈空间

    std::function<void()> m_cb; // 运行函数
    bool m_runInScheduler;
```



由于该类过程大致为：

```sql
| 主协程        | 协程调度器
       | 协程A  | 协程调度器
| 主协程        | 协程调度器
       | 协程B  | 协程调度器
```

在这里存在两种协程调度模式：

- 参与调度器，有调度器统一管理协程的切换
- 不参与调度器，直接与主线程切换上下文

```cpp
// 当前正在运行的协程
static thread_local Fiber *t_fiber = nullptr;
// 主协程，管理声明周期
static thread_local std::shared_ptr<Fiber> t_thread_fiber = nullptr;
// 调度协程，管理指针访问
static thread_local Fiber *t_scheduler_fiber = nullptr;
```

整个流程：

1. **主协程创建**：线程启动时创建主协程，保存主线程上下文。

2. **用户协程创建**：通过分配栈空间和初始化上下文创建用户协程。

3. **协程运行**：

   - `resume` 将协程切换到 **RUNNING**，执行任务。

   - 任务完成后，协程状态变为 **TERM**，调用 `yield` 切换回调用方上下文。

4. **协程管理**：支持通过调度器统一管理协程或直接与主线程切换。

这里目前未能体现主协程和调度线程的差别，具体需要等待下一个部分



- [ ] 协程类，写一段话
- [ ] debug
- [ ] test
- [ ] 调度池和线程池比较



## 协程调度

一个线程只有一个协程，一个协程类中会包含三个协程，分别是主协程（main），调度协程和任务协程。其中任务协程是由协程类自主创建，主协程和调度协程都是静态变量，在多种类中其实只存在一个实体。

协程调度致力于封装一些操作，因为调度协程本身需要创建协程，协程任务的执行顺序，如何利用多线程或者调度协程池保证效率，在协程任务结束之后也需要停止调度器释放资源。如果建立一个scheduler类封装这些操作，那么为用户开放的仅仅只有**启动线程池，关闭线程池，添加任务**三种操作了。

> 引用：（来源：代码随想录）
>
> 调度器内部维护一个任务队列和一个调度线程池。开始调度后，线程池从任务队列里按顺序取任务执行。调度线程可以包含caller线程。当全部任务都执行完了，线程池停止调度，等新的任务进来。添加新任务后，通知线程池有新的任务进来了，线程池重新开始运行调度。停止调度时，各调度线程退出，调度器停止工作。
>
> 1. main函数主协程运行，创建调度器
> 2. 仍然是main函数主协程运行，向调度器添加一些调度任务
> 3. 开始协程调度，main函数主协程让出执行权，切换到调度协程，调度协程从任务队列里按顺序执行所有的任务
> 4. 每次执行一个任务，调度协程都要让出执行权，再切到该任务的协程里去执行，任务执行结束后，还要再切回调度协程，继续下一个任务的调度
> 5. 所有任务都执行完后，调度协程还要让出执行权并切回main函数主协程，以保证程序能顺利结束。

**主协程和调度协程的差异**

| **特性**         | **主协程**                     | **调度协程**               |
| ---------------- | ------------------------------ | -------------------------- |
| **初始化方式**   | 在线程启动时自动初始化         | 调度器初始化时创建         |
| **标识**         | `t_thread_fiber`               | `t_scheduler_fiber`        |
| **上下文栈空间** | 使用线程自身栈                 | 使用独立分配的栈           |
| **主要功能**     | 恢复线程的原始上下文，终止线程 | 切换并调度用户协程         |
| **运行逻辑**     | 不主动运行任何任务逻辑         | 包含协程调度的逻辑         |
| **让出目标**     | 让出后通常切换到主线程运行     | 让出后切换回调度器逻辑     |
| **用途**         | 管理线程范围内的根协程         | 管理其他协程，选择运行协程 |



> 为什么有主协程，调度协程，任务协程三种设置



> 协程的基础知识

## 协程IO

## 定时器

## hook



好的，以下是每个函数的含义，以及各个参数的解释：

### 1. `sleep` 和 `usleep`，`nanosleep`

#### 1.1 `sleep(unsigned int seconds)`

- **含义**：使当前进程暂停执行指定的秒数。期间进程不执行任何操作，直到时间到期或被信号中断。
- 参数
  - `seconds`：要暂停的时间，单位为秒。
- **返回值**：返回剩余的睡眠时间（秒）。如果被信号中断，返回的值是剩余的时间。

#### 1.2 `usleep(useconds_t usec)`

- **含义**：使当前进程暂停执行指定的微秒数。
- 参数
  - `usec`：要暂停的时间，单位为微秒。
- **返回值**：成功时返回0。如果发生错误，返回-1，并设置 `errno`。

#### 1.3 `nanosleep(const struct timespec *req, struct timespec *rem)`

- **含义**：使当前进程暂停执行指定的时间，精度为纳秒。
- 参数
  - `req`：指定暂停的时间。类型为 `struct timespec`，其中 `tv_sec` 表示秒数，`tv_nsec` 表示纳秒数。
  - `rem`：如果系统调用被信号中断，返回未完成的时间。这个参数是输出参数，存放剩余时间。
- **返回值**：成功时返回0，如果被信号中断，返回-1，并设置 `errno`。

### 2. Socket 函数

#### 2.1 `socket(int domain, int type, int protocol)`

- **含义**：创建一个新的套接字，用于网络通信。
- 参数
  - `domain`：协议族，指定套接字的协议类型。例如：`AF_INET`（IPv4地址族），`AF_INET6`（IPv6地址族），`AF_UNIX`（Unix本地通信）。
  - `type`：套接字类型，指定套接字的通信方式。例如：`SOCK_STREAM`（流式套接字，TCP连接），`SOCK_DGRAM`（数据报套接字，UDP协议）。
  - `protocol`：协议类型，通常设为0，表示选择与套接字类型相关的默认协议。
- **返回值**：返回套接字的文件描述符（`sockfd`），失败时返回-1，并设置 `errno`。

#### 2.2 `connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)`

- **含义**：连接一个套接字到指定的地址，通常用于客户端发起连接。
- 参数
  - `sockfd`：创建的套接字描述符。
  - `addr`：指向 `struct sockaddr` 的指针，包含目标地址的信息。
  - `addrlen`：`addr` 的长度。
- **返回值**：成功时返回0，失败时返回-1，并设置 `errno`。

#### 2.3 `accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)`

- **含义**：接受一个传入的连接请求，通常由服务器调用。
- 参数
  - `sockfd`：监听套接字的文件描述符，通常是通过 `socket` 创建并调用 `bind` 和 `listen` 后的套接字。
  - `addr`：输出参数，返回客户端的地址信息，类型为 `struct sockaddr`。
  - `addrlen`：输入输出参数，表示 `addr` 的大小。调用时传入 `socklen_t` 类型的值，返回时会修改为实际填充的字节数。
- **返回值**：成功时返回一个新的套接字文件描述符，表示与客户端的连接；失败时返回-1，并设置 `errno`。

### 3. 读操作

#### 3.1 `read(int fd, void *buf, size_t count)`

- **含义**：从指定文件描述符 `fd` 中读取数据。
- 参数
  - `fd`：文件描述符，通常是文件、管道、设备或套接字。
  - `buf`：指向缓冲区的指针，数据将被读取到该缓冲区。
  - `count`：要读取的字节数。
- **返回值**：成功时返回读取的字节数，失败时返回-1，并设置 `errno`。

#### 3.2 `readv(int fd, const struct iovec *iov, int iovcnt)`

- **含义**：从指定文件描述符读取多个缓冲区的数据（`read` 的扩展）。
- 参数
  - `fd`：文件描述符。
  - `iov`：指向 `struct iovec` 数组的指针。`struct iovec` 结构包含两个字段：`iov_base`（指向数据缓冲区的指针）和 `iov_len`（缓冲区的大小）。
  - `iovcnt`：`iov` 数组中的元素数量。
- **返回值**：成功时返回读取的字节数，失败时返回-1，并设置 `errno`。

#### 3.3 `recv(int sockfd, void *buf, size_t len, int flags)`

- **含义**：从套接字接收数据，类似于 `read`，但用于网络套接字。
- 参数
  - `sockfd`：套接字的文件描述符。
  - `buf`：数据将被存储的缓冲区。
  - `len`：要接收的最大字节数。
  - `flags`：接收操作的选项，通常是0，也可以设置为其他值，例如 `MSG_PEEK`。
- **返回值**：成功时返回接收的字节数，失败时返回-1，并设置 `errno`。

#### 3.4 `recvfrom(int sockfd, void *buf, size_t len, int flags, struct sockaddr *src_addr, socklen_t *addrlen)`

- **含义**：从套接字接收数据，并获取发送者的地址，通常用于 UDP 套接字。
- 参数
  - `sockfd`：套接字的文件描述符。
  - `buf`：数据将被存储的缓冲区。
  - `len`：要接收的最大字节数。
  - `flags`：接收操作的选项，通常是0。
  - `src_addr`：输出参数，接收方的地址信息。
  - `addrlen`：输入输出参数，表示 `src_addr` 的大小，调用时传入，返回时更新为实际地址大小。
- **返回值**：成功时返回接收的字节数，失败时返回-1，并设置 `errno`。

#### 3.5 `recvmsg(int sockfd, struct msghdr *msg, int flags)`

- **含义**：接收多个消息或者带有复杂头部的消息，通常用于需要复杂协议的套接字（如 ICMP）。
- 参数
  - `sockfd`：套接字的文件描述符。
  - `msg`：指向 `msghdr` 结构的指针，`msghdr` 结构包含了消息的各个部分，如消息数据和头部信息。
  - `flags`：接收操作的选项。
- **返回值**：成功时返回接收到的字节数，失败时返回-1，并设置 `errno`。

### 4. 写操作

#### 4.1 `write(int fd, const void *buf, size_t count)`

- **含义**：向文件描述符写入数据。
- 参数
  - `fd`：文件描述符。
  - `buf`：指向缓冲区的指针，包含要写入的数据。
  - `count`：要写入的字节数。
- **返回值**：成功时返回写入的字节数，失败时返回-1，并设置 `errno`。

#### 4.2 `writev(int fd, const struct iovec *iov, int iovcnt)`

- **含义**：向文件描述符写入多个缓冲区的数据（`write` 的扩展）。

- 参数

  ：

  - `fd`：文件描述符。
  - `iov`：指向 `struct iovec` 数组的指针。
  - `iovcnt`：`iov` 数组中的元素数量。

- **返回值**：成功时返回写入的字节数，失败时返回-1，并设置 `errno`。

#### 4.3 `send(int sockfd, const void *buf, size_t len, int flags)`

- **含义**：通过套接字发送数据，通常用于 TCP 连接。
- 参数
  - `sockfd`：套接字的文件描述符。
  - `buf`：要发送的数据。
  - `len`：数据的字节数。
  - `flags`：发送操作的选项。
- **返回值**：成功时返回发送的字节数，失败时返回-1，并设置 `errno`。

#### 4.4 `sendto(int sockfd, const void *buf, size_t len, int flags, const struct sockaddr *dest_addr, socklen_t addrlen)`

- **含义**：通过套接字发送数据，通常用于 UDP 协议。
- 参数
  - `sockfd`：套接字的文件描述符。
  - `buf`：要发送的数据。
  - `len`：数据的字节数。
  - `flags`：发送操作的选项。
  - `dest_addr`：目标地址信息。
  - `addrlen`：目标地址的大小。
- **返回值**：成功时返回发送的字节数，失败时返回-1，并设置 `errno`。

#### 4.5 `sendmsg(int sockfd, const struct msghdr *msg, int flags)`

- **含义**：发送带有复杂消息头的数据，通常用于需要复杂协议的套接字（如 ICMP）。
- 参数
  - `sockfd`：套接字的文件描述符。
  - `msg`：指向 `msghdr` 结构的指针。
  - `flags`：发送操作的选项。
- **返回值**：成功时返回发送的字节数，失败时返回-1，并设置 `errno`。

### 5. 文件描述符操作

#### 5.1 `close(int fd)`

- **含义**：关闭文件描述符，释放相关资源。
- 参数
  - `fd`：文件描述符。
- **返回值**：成功时返回0，失败时返回-1，并设置 `errno`。

### 6. Socket 控制函数

#### 6.1 `fcntl(int fd, int cmd, ... /* arg */)`

- **含义**：操作文件描述符的各种属性，如设置非阻塞模式等。
- 参数
  - `fd`：文件描述符。
  - `cmd`：命令，指定执行的操作（例如 `F_GETFL` 获取文件状态标志，`F_SETFL` 设置文件状态标志）。
  - `arg`：命令的附加参数，根据命令不同，可能是一个整数或指针。
- **返回值**：操作的结果，具体取决于命令和操作。

#### 6.2 `ioctl(int fd, unsigned long request, ...)`

- **含义**：控制设备或套接字的行为。
- 参数
  - `fd`：文件描述符。
  - `request`：控制命令，指定要执行的操作。
  - `...`：命令的附加参数。
- **返回值**：成功时返回0，失败时返回-1，并设置 `errno`。

#### 6.3 `getsockopt(int sockfd, int level, int optname, void *optval, socklen_t *optlen)`

- **含义**：获取套接字的选项值。
- 参数
  - `sockfd`：套接字的文件描述符。
  - `level`：协议层级，指定选项的协议层级（例如 `SOL_SOCKET`）。
  - `optname`：选项名称，指定要获取的选项。
  - `optval`：输出参数，接收选项的值。
  - `optlen`：输入输出参数，表示 `optval` 的大小，调用时传入，返回时更新为实际大小。
- **返回值**：成功时返回0，失败时返回-1，并设置 `errno`。



# 参考

1. 代码随想录协程库项目精讲

