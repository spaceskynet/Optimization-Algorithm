# Armijo 不精确一维搜索求解 RosenBrock 函数近似步长
# Author:SpaceSkyNet
using LinearAlgebra
const global ρ = 0.5 # 放缩因子

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
    Armijo(times, σ, α, x_pre, p_pre, f_pre, gkT_pk) -> Float64, Int64

Armijo 不精确一维搜索, 输出对应步长 `α` 和迭代步数
"""
function Armijo(times::Int64, σ::Float64, α::Float64, x_pre::Vector{Float64}, p_pre::Vector{Float64}, f_pre::Float64, gkT_pk)
    x_now = x_pre + α * p_pre
    f_now = f(x_now)
    println("Step $times:")
    println("\tx[k] = $x_pre', f[k] = $f_pre")
    println("\tx[k+1] = $x_now', f[k+1] = $f_now")
    L = f_pre - f_now
    R = -1 * σ * α * gkT_pk
    if L >= R
        println("\tf[k]-f[k+1] = $L >= -σ * α * g[k]T*p[k] = $R")
        println("\tNow α = $α, Meet termination condition, stop.")
        return α, times + 1
    end
    println("\tf[k]-f[k+1] = $L < -σ * α * g[k]T*p[k] = $R")
    println("\tNow α = $α, continue.")
    return Armijo(times + 1, σ, α * ρ, x_pre, p_pre, f_pre, gkT_pk)
end

# σ = rand() / 2 # 随机产生(0, 0.5)的小数
# 随机产生(0.1, 0.5)的小数, 实际测试发现对该问题大概在 σ < 0.1 时，算法会在 α = 0.25 时终止
σ = rand() * (0.5 - 0.1) + 0.1 
x_k = [0.0; 0.0]
p_k = [1.0; 0.0]
α, times = Armijo(0, σ, 1.0, x_k, p_k, f(x_k), g(x_k)' * p_k)
println("Final:\nα = $α,\nσ = $σ,\n$times steps in total.")


