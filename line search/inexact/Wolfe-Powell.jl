# Wolfe-Powell 不精确一维搜索求解 RosenBrock 函数近似步长
# Author:SpaceSkyNet
using LinearAlgebra
const global ρ = 0.5 # 放缩因子
global a, b = 0, Inf # 区间 [a, b]

"""
    f(x::Vector{Float64}) -> Float64

RosenBrock函数, 输出对应变量的值
"""
function f(x::Vector{Float64})
    return Float64(100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2)
end

"""
    g(x::Vector{Float64}) -> Vector{Float64}

RosenBrock函数的梯度函数, 输出对应变量梯度的值
"""
function g(x::Vector{Float64})
    return [-400 * (x[2] - x[1]^2) * x[1] - 2 * (1 - x[1]); 200 * (x[2] - x[1]^2)]
end

"""
    Wolfe_Powell(times, μ, α, x_pre, p_pre, f_pre, gkT_pk) -> Float64, Int64

Wolfe_Powell 不精确一维搜索, 输出对应步长 `α` 和迭代步数
"""
function Wolfe_Powell(times::Int64, μ::Float64, σ::Float64, α::Float64, x_pre::Vector{Float64}, p_pre::Vector{Float64}, f_pre::Float64, gkT_pk)
    global a, b
    x_now = x_pre + α * p_pre
    f_now = f(x_now)
    println("Step $times:")
    println("\tx[k] = $x_pre', f[k] = $f_pre")
    println("\tx[k+1] = $x_now', f[k+1] = $f_now")
    L = f_pre - f_now
    R =  -1 * μ * α * gkT_pk # 第一条件形式
    if L >= R
        println("\tf[k] - f[k+1] = $L >= -μ * α * g[k]T*p[k] = $R")
        println("\tMeet the first condition, continue.")
        L = abs(g(x_now)' * p_pre)
        R = -1 * σ * gkT_pk # 第二强条件形式
        if L <= R
            println("\tg[k+1] * p[k] = $L <= σ * g[k]T*p[k] = $R")
            println("\tNow α = $α, [a, b] = [$a, $b], Meet the second condition, stop.")
            return α, times + 1
        else
            println("\tg[k+1] * p[k] = $L > σ * g[k]T*p[k] = $R")
            println("\tNow α = $α, [a, b] = [$a, $b], Not meet the second condition, continue.")
            a = α
            return Wolfe_Powell(times + 1, μ, σ, min(α / ρ, (α + b) * ρ), x_pre, p_pre, f_pre, gkT_pk)
        end
    else
        println("\tf[k] - f[k+1] = $L < -μ * α * g[k]T*p[k] = $R")
        println("\tNow α = $α, [a, b] = [$a, $b], Not meet the first condition, continue.")
        b = α
        return Wolfe_Powell(times + 1, μ, σ, (α + a) * ρ, x_pre, p_pre, f_pre, gkT_pk)
    end
end
#=
随机产生(0.1, 0.5)的小数和产生(0.5, 1)的小数, 实际测试发现对该问题大概在 μ < 0.1 时，算法会在 α = 0.25 时终止
或者
在 σ 相对 μ 较小时，会无法满足第二条件而陷入死循环
实际测试发现对该问题大概在 σ < 0.5 时，算法会在Step 3满足第一条件而不满足第二条件，在Step 4时才终止
=#
μ = rand() * (0.5 - 0.1) + 0.1 
σ = rand() * (1 - 0.5) + 0.5
# μ = 0.12488239591026568, 0.27771434370696935
# σ = 0.4701966514809402, 0.49251788333943214
x_k = [0.0; 0.0]
p_k = [1.0; 0.0]
α, times = Wolfe_Powell(0, μ, σ, 1.0, x_k, p_k, f(x_k), g(x_k)' * p_k)
println("Final:\nα = $α,\nμ = $μ, σ = $σ,\n$times steps in total.")


