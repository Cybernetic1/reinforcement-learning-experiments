# RL with Auto-Encoder

The following are my personal notes, may seem disorganized and incomprehensible.

<img src="minimal-RL.png" width="150"/>

<img src="RL-with-autoencoder.png" width="170"/>

RL 產生的 δx 是否应放进 自编码器？  
自编码器 的 x 明显是属于 RL state 的一部分  
似乎 二者的内容应该分享  
但 二者是如何融合？  
RL 根据什么给出 δx？每个 x 有价值函数  
AE 根据的是 reconstruction error  
但二者是否可以 同时运行而没有抵触？  
每个 algorithm 更新的其实是 weights 而已  
但状态的改变会否影响算法的收敛？  
在资料有限的情况下，  
算法收敛到能准确地解释资料的程度  

但如果将这架构 应用到 井子棋上则好像不行？  
问题是预测 TTT 的状态  
然后状态 再推导出行动

<img src="RL-with-autoencoder-TTT.png" width="300"/>

1. 有个问题是：RL 输出给 AE 有什么意思？  
因为 RL 其实直接输出下一步，  
而不是什么隐状态。除非令 RL = 多步逻辑！  
是可以的，但有些隐状态用来预测下一步  
似乎不合适。或者说：RL 的隐状态用来  
预测世界，总是合适的？
2. 另一个问题： 多步 RL 如何达成？  
辨别 隐状态 vs 动作状态  
似乎要增加 状态空间的长度。
3. 还有就是 多步逻辑的奖励问题  
价值函数是 associated with 全局状态  
但奖励可以是关于局部状态的  
可不可以 建构 状态的 **分拆**？

这种分拆似乎很重要。  
可不可以将 x 和 δx 的表述 显式地 写进 RL 里？  
例如 将 状态 = δx？那么全状态又是什么？

<img src="state-with-delta.png" width="400"/>

全状态 也是一个状态，所以应该做的是 状态的 **细致化**。

全局上，我们关心的是 状态 x 的价值，那么 V(x + δx) 能否表达成 V(x) 的推导？

1. 很明显 x + δx 是 x 的 下一个 reachable 状态。 
2. x + δx 的价值是外在定义的，但它跟 x + δx 的形式无关
3. 它是 action δx 导致的，这似乎跟动力学方程有关
4. 这是一种 状态 algebra 结构，原来的 RL 推演中没有
5. algebra of states 是如何影响 RL 的 formulation？  
似乎最简单的答案就是 定义 R(δx), 即每个 **命题** 的奖励。  
而 x 的奖励是 根据 δx 决定的。 甚至反而更直观、更易处理。

## RL 跟 AE 的 互相干涉

1. RL 干涉 AE：  
AE 的中部突然出现外来的 tokens 会否影响其收敛？
RL 导致出现的是：  
a）关于事实的思考，  
b）关于行动的思考  
然后用这些来预测世界，合适吗？  
 (a) 还可以，但 (b) 似乎有些牵强？  
（例如期待某人被暗杀、所以关注军队有否叛变等各种蛛丝马迹）

2. AE 干涉 RL：  
RL 的状态中突然出现外来的 δx 会否影响 RL 的收敛？  
应该不会，因为**状态从来是随着世界改变的**。

我说明了可以收敛，但没有说明 有没有帮助？  
AE → RL 肯定是有帮助的  
RL → AE 或许可以用 special marker 标记  
后者或许可以帮助 AE 涌现更多智慧

<img src="state-with-intermediates.png" width="600"/>

为方便起见 设 N=9

Other than the board vector, we need an auxiliary store of propositions.

But it may be different from the board vector.

Each proposition is a discrete value from { 0...8 },
so 9 propositions has 9<sup>9</sup> combinations but with redundancy.

If not counting repeats, it is <sub>9</sub>C<sub>1</sub> + <sub>9</sub>C<sub>2</sub> +... + <sub>9</sub>C<sub>9</sub> = 2<sup>9</sup>.

Two questions:

1. Shall we allow deleting (negating) a proposition?

2. Forgetting.  Perhaps we should use a list to implement this.
In our simple situation we can actually have "permanent" memory and
learning would still be OK.

究竟 多步逻辑 是不是有需要呢？  
以下棋作比喻，V 包含了每个思维的价值  
但思维与下棋是不是完全相似的？  
在思维空间里 每个状态 映射到 所有 δx 的概率分布  
状态 x 是包含很多个命题的，但这个 映射不能被分拆  
（这个映射的复杂性，适宜用深度神经网络处理）  
现在问题是 next thought  
如果直接输出 action 就行，为什么要 thoughts？  
thoughts 的奖励机制是很复杂的  
它涉及到 资讯压缩的 gains  
现在问题是：漫无目的地产生 thoughts，  
奖励机制 不明显。  
压缩的 loss function 似乎就是 world reconstruction loss.  
问题是有没有坚定不拨的 学习原则？  
既然 thought is action 则它必然符合 Bellman 条件  
但它似乎太 “free”，有没有其他的约束？  
「**双重约束**」是否在理论上一致？  

其实 RL 的 world model 也是根据 truth 而学习，而不是 Bellman  
那么 AE 也是根据 truth 而学习。 是不是根本没有矛盾？  

其实 GPT 的做法就是将 thoughts 外在化，变成自然语言  

## Looping

算法似乎不喜欢用 looping，可以怎样解决？  
如果它可以输出某个动作，它不妨直接输出那个动作。  
我的假设认为（而且也在一定程度上证实了）算法不足够表达所有的 maps

似乎有几个方法可以解决：  
1.  只容许一步 intermediate  
    但是否要 aggregate 输出 还是 trancate 输出?  

x1 + 0 ↦ δx  (有可能被奖励, 因为下一句)  
x1 + δx ↦ y  会被奖励  
x1 + 0 ↦ y  也是正确的, 而且有可能被学习  
问题是 δx 有没有其他功能?

x2 + 0 ↦ δx  (这个可能性有多大?)  
它表示 δx 代表了一个 x1, x2 共有的特征
x2 + δx ↦ y2  (这个可能性有多大?)  

## Looping 失败

假设 NN 可以表示 10个 maps  
理论上 NN.NN 可以表示 10x10 = 100 个 maps  
因为中间的 状态 是"浮动"的

还是不明白 中间 maps 被淘汰的原因....  
会不会是 A ↦ B 然后 B ↦ D which is losing move?  
因为 B 导致了其他 losing moves 所以 B 被淘汰?

然后还有个问题是: 在状态 x 里, 命题的个数 n 增大时, 个别命题的影响力 似乎越来越小.  也就是说, NN 作为 consequence operator map 并不太符合 逻辑 rules 推导的性质 (通常是从少量的前提下推出结论).  但其实也没有理由认为 NN 一定不能学习这种形状的 mapping.

完全不明白 thoughts 被淘汰的原因....  
根据 RL 的理论, 它会遍历所有的状态空间  
多出的输入 为什么没有用?  
因为根据前半部 NN 的分析, 它的资讯已经 **不足够**  
所以 thoughts 只能反映那些不足够的资讯?  
(这似乎是最合理的解释)

另一个想法是:  thoughts 和 inputs 的语义不是 **同质** 的?  
其实这好像不太重要  
重要的是 NN 有没有能力模拟 logic rules 特别是 "少量前提特性"

### Sym NN 试验到此为止

效果不太好, 估计是因为 Sym NN 的结构 不太像 逻辑 rules.  
Transformer 应该更接近 逻辑 rules 的结构.  

## Transformer

在这场景下 Transformer 是否有用?  
Transformer 对应的 recurrence 是什么?  
它比 RNN 好, 意思是 RNN per token recurse  
但我现在是 Transformer 的 action recurse  
但我现在考虑的是 states 的重复....  
似乎后者并不需要 多一重的 Transformer,  
因为 state 与 next state 之间是有 δ 的关系

## RL 与 LLM 「相撞」的问题

这仍然是最重要的问题。 RL 负责「思考」，但 LLM 亦有此功能。  
为了预测未来，LLM 必需有某种思考能力，但这跟 RL 的功能重叠了。  
但 RL 的结构是循环的，我们想用它替代 LLM？  
那么 RL 是否可以完全取代 LLM？  
或者说： LLM 学习 p，RL 学习 π，为什么 π 和 p 会相撞？  
RL 的目标是由 价值 / 奖励 决定的，  
如果 RL 的目标就是 解释世界，则 RL 和 LLM 目标一样 而算法不同  
但更一般地说，RL 的价值不只是解释世界  
它有 human interests.

重要的问题是：RL 和 LLM 共享状态  
承接上面的论述, 问题是 LLM 能不能接受 RL 干扰 它的状态?  
似乎是可以的, 那么 LLM 的目标仍然是 学习 p  

## RL 能否高效率地学习？

关键是：RL 能否学习到更好的逻辑规则（相对于 LLM 而言）？  

## 受 Looped Transformer 启发之后

So, what's new?  它似乎提供了一些训练的细节？  
误差的计算是根据 expected output token 的 squared error  
从 embedding 假设看来，这似乎还 make sense

这跟我井字棋那边的失败有什么关系？  
我的做法是 状态空间中有 intermediate states  
Loop TRM 的做法是： 必然 loop T 次，  
计算 mean( sum of square losses )  
而且，它是不是作为 auto-encoder 似乎也是关键的？  
因为在我的 井字棋 实验里，TRM 是 δx operator  
因为 Loop TRM 的 loss 是整个 prompt 序列的预测 loss  
换句话说： 会不会 TRM 其实是特别为了 auto-encoding 的工作而优化的？  
如果 TRM 可以学习 next x 的分布，为什么它不可以学习 δx 的分布？

δx 的输出似乎包含了 logic attention  
同一个状态 可以输出 不同的 δx，这似乎不仅是概率的差异  
它似乎需要更多的時間 explore search space？  

## 现在有3个问题

1. 为什么 Looped TicTacToe 拿不到奖励？
2. Architecture (for TicTacToe) -- RL 的输出究竟是 t 还是 t + 1？ 
3. Search efficiency when looped. Is there a difference?

**Q2:**  如果 RL 输出的是 t + 1, 则它直接输出到 output？  
但又好像不是，因为 RL 输出的是隐状态。  
其实 t 和 t+1 似乎没有良好定义。  
在大脑中，似乎是有某种天然的节奏吧？  
问题是：什么才算是 t 的推论，什么算是对 t+1 的预测？  
会不会是两套不同的 mappings？  
问题在于它们 verification 的方法。   
当下状态 产生新的推论，但仍然属于 当下的。  
然后 状态 t **预测** 状态 t+1.  
但 verify 仍然是针对 t+1 而言的，理由很简单，因为 t 是**已知**的了。  
预测 map 和 推理 map 似乎不是同一个 map.
g
