---
layout:       post
title:        "算法专题：算法和数学知识"
author:       "Sutdown"
header-style: text
catalog:      true
tags:
    - algorithm
    - C++
---


注：来源Acwing算法基础课。

#### [868. 筛质数 - AcWing题库](https://www.acwing.com/problem/content/870/)

> 给定一个正整数 n，请你求出 1∼n中质数的个数。

首先，暴力做法，对每一个数判断是不是质数，然后res记录个数，由于每一次判断是不是质数需要一次for循环，因此时间复杂度为O（n^2），意料中的超时了。

```C++
#include<iostream>
using namespace std;

bool isPrime(int x) {
	if (x == 1)return false;
	for (int i = 2; i <= x / i; i++) {
		if (x % i == 0)return false;
	}
	return true;
}

int main() {
	int n;
	cin >> n;
	int res=0;
	for(int i=2;i<=n;i++){
		if (isPrime(i))res++;
	}
	cout << res << endl;
	return 0;
}
```

所以，对于这道题，我们换一种思路，筛选出来质数并不一定要找出所有质数，将合数排除也可以得到质数。

所以，从头至尾，记录下所有的质数的个数，同时将所有质数倍数的合数标记为1不计入。由于第i个如果是合数，那么一定由比其小的质数的因数，该数会被标记成1，所以一定不会计入个数，而若第i个是质数，那么s[i]仍然是0，会计入个数，因此遍历所有就能得到结果。

```
#include<iostream>
using namespace std;
const int N = 1e6 + 10;

int s[N];
int n, cnt;

int main() {
	cin >> n;

	//1.1既不是质数也不是合数，因此从2开始判断。
	//2.合数都可以由一系列质数相乘得到
	//3.st中默认为0，因此如果是质数我们将其标记为1；
	for (int i = 2; i <= n; i++) {
		if (!s[i])cnt++;
		for (int j = 2; j <= n / i; j++) {
			s[j * i] = 1;
		}
	}
	cout << cnt << endl;
	return 0;
}
```



#### [869. 试除法求约数 - AcWing题库](https://www.acwing.com/problem/content/871/)

> 给定 n个正整数 ai，对于每个整数 ai，请你按照从小到大的顺序输出它的所有约数。

求约数，先给大家看看我的垃圾代码。就是简单的是约数就加入数组中，然后排序。

1）2乘以1e9这么大的数是怎么用数组申请这么大空间的，一个根本不能，一个没有必要！！！

答：容器啊容器啊，你忘了vector吗，vector相比普通数组可以动态的先申请部分空间，再如果不够自己会重新申请，这比普通数组不知道好了多少。

2）vector输出结果时会有重复元素如何解决？
如果对加入的数进行判断在数组中是否有重复是一件很浪费时间的事情，所以不建议。这时候突然发现了set容器，它有两个特点完美的契合了这道题。一是可以自动去除重复元素；二是可以自动排序，sort都省了！！快乐

附上两个链接，对vector和set的简单介绍

[Vector 简介和优缺点_vector的缺点-CSDN博客](https://blog.csdn.net/mercy_ps/article/details/81098577)

[深入理解C++ STL库中的set容器-CSDN博客](https://blog.csdn.net/YoyoHuzeyou/article/details/132184083#:~:text=4.set容器的特性 1 唯一性：set容器中的元素是唯一的，不会出现重复的元素。 2,有序性：set容器中的元素按照一定的顺序进行排序，默认是升序。 可以通过自定义比较函数实现自定义排序。 3 动态增删：set容器支持动态地增加和删除元素，同时保持有序性。)

```
#include<iostream>
#include<algorithm>
using namespace std;
const int N = 2 * 1e9 + 10;

int cnt = 0;
int a[N];

int main() {
	int n;
	cin >> n;

	while (n--) {
		int x;
		cin >> x;
		//1.如何找约数。
		//可以从小到大直接找就行
		//2.如何从小到大输出。
		//排序？会有点慢，不知道能不能过。

		for (int i = 1; i <= x / i; i++)
			if (x % i == 0) {
				a[cnt++] = i;
				a[cnt++] = n / i;
			}
		sort(a, a + cnt);

		for (int i = 0; i <= cnt; i++) {
			cout << a[i] << ' ' ;
		}
		cout << endl;
	}

	for (int i = 0; i <= cnt; i++) {
		cout << a[i] << ' ' ;
	}
	return 0;
}
```

来看看接下来的修改版代码，成功AC；

```C++
#include<iostream>
#include<set>
using namespace std;

int main() {
	int n;
	cin >> n;

	while (n--) {
		int x;
		cin >> x;
	
		set<int> a;

		for (int i = 1; i <= x / i; i++)
			if (x % i == 0) {
				a.insert(i);
				a.insert(x / i);
			}

		for (auto x : a) cout << x << ' ';
		cout << endl;
	}

	return 0;
}
```



#### [870. 约数个数 - AcWing题库](https://www.acwing.com/problem/content/description/872/)

> 给定 n个正整数 ai，请你输出这些数的乘积的约数个数，答案对 1e9+7 取模。

第一思路，直接相乘积再求约数个数。不过相乘可能超过int类型，求约数个数也力不从心，所以应该还是逐个分析。

~~第二反应，有两个直觉思路，都是用set，自动去重和排序真好用~~

~~1）将所有的约数保存到set容器，再将其两两相乘。~~

~~2）用几个数的乘积去处于所有的约数。~~

这两种都是属于有点麻烦的，重新看如何判断约数个数这个问题。约数可以写成几个质数的底数的指数次方
$$
S=a^n*b^m...(个数为（n+1）*（m+1）...)
$$
相乘的形式，每个底数都是不同的，然后对于每个a的n次方，a可以取0-n有n+1种方式，同理后面为m+1，它们各部分相乘的积也一定是S的因数，所以个数就是指数+1的乘积了。

```c++
#include<iostream>
#include<unordered_map>
using namespace std;
const int mod=1e9+7;

int main() {
	int n;
	cin >> n;
	
	long res=1;//用long，防止数据过大越界
	unordered_map<int, int> a;
	
	while (n--) {
		int x;
		cin >> x;

		for (int i = 2; i <= x / i; i++)
			while (x % i == 0) {//指数的次数
				a[i]++;
				x = x / i;//对x因数乘积的判断
			}
		if (x > 1)a[x]++;
	}

	for (auto i : a)res = res * (1 + i.second) % mod;
	cout << res << endl;
	return 0;
}
```



#### [874. 筛法求欧拉函数 - AcWing题库](https://www.acwing.com/problem/content/876/)

> 给定一个正整数n，求1∼n中每个数的欧拉函数之和。

惯例，先看看TLE（Time Limit Exceeded）的算法。注解放在算法中了。

```C++
#include<iostream>
using namespace std;
const int N = 110;

//判断是否是质数，指数的欧拉函数就是其本身-1；
bool prime(int n) {
	for (int i = 2; i <= n / 2; i++) {
		if (n % i == 0)return false;
	}
	return true;
}

//欧拉函数的判断方法，按照欧拉函数的定义写出来的代码
//链接见：https://zhuanlan.zhihu.com/p/151756874
int oula(int n) {
	if (n == 1)return 1;

	if (prime(n))return n - 1;

	int p[N], cnt = 0, res = n;
	for (int i = 2; i <= n / i; i++) {
		if (n % i == 0) {
			p[cnt++] = i;
		}
		while (n % i == 0) {
			n = n / i;
		}
	}
	if (n > 1)p[cnt++] = n;
	for (int i = 0; i < cnt; i++) {
		//要先除再乘，避免溢出
		res = res / p[i] * ( p[i] - 1 );
	}
	return res;
}

int main() {
	int n;
	cin >> n;

	int res = 0;
	for (int i = 1; i <= n; i++)
		res += oula(i);

	cout << res << endl;
	return 0;
}
```

orz，看了大佬的题解，这是要多么深厚的数学基础，放个链接[AcWing 874. 筛法求欧拉函数 - AcWing](https://www.acwing.com/solution/content/3952/)

捋一捋思路，本质上是数学题。



#### [875. 快速幂 - AcWing题库](https://www.acwing.com/problem/content/877/)

> 给定n组 ai，bi，pi，对于每组数据，求出ai*bimodpi的值。

第一反应，暴力做法，a的b次幂要怎么求，要怎么取模，如果想直接把a的b次幂表示出来，所花的时间必然会过长导致超时，所以猜测这题的考虑点在如何在求a的b次幂时降低它的时间复杂度。也就暂时不考虑取模。

求a的b次幂时，本来是要将a乘b次，那么如何降低乘的次数呢，有一个性质，所有的十进制数是可以由二进制数表示的，那么a需要乘的字数自然就降低了。二进制中每位要么是1要么是0，不仅降低了b次的次数，每次的运算量也不大，完美解决。

这个和01背包也有一点点相似，都是0和1的形式。

```c++
#include<iostream>
using namespace std;

long long Quickmi(long long a, int b, int p) {
	long long res = 1;
	while (b) {
		if (b & 1) {
			//这个a指的是b中最后一位为1时a的值
			//a第一次就是a，往后就是a的自身的平方
			res = res * a % p;
		}
		b >>= 1;
		a = a * a % p;
	}
	return res;
}
int main() {
	int n;
	cin >> n;
	while (n--) {
		cin.tie(0);
		ios::sync_with_stdio(false);
		int a, b, p;
		cin >> a >> b >> p;
		cout << Quickmi(a, b, p) << endl;
	}

	return 0;
}
```



#### [877. 扩展欧几里得算法 - AcWing题库](https://www.acwing.com/problem/content/879/)

> 给定n对正整数 ai,bi，对于每对数，求出一组 xi,yi，使其满足 ai×xi+bi×yi=gcd(ai,bi)。

