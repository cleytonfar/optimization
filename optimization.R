#============================================================================
# FILE: optimization.R
#
# LANGUAGE*: R
#
# DESCIPTION: This file contains the necessary routine to perform 
#             optimization in the following examples:
#                - minimize the function: x*exp(-x^2 - y^2) + (x^2 + y^2)/20
#                - maximize the log-likelihood for linear regression 
#                - maximize the log-likelihood for logit regression
#
# AUTHOR: Cleyton Farias
#
# DATE: June 21, 2017
#
# LAST MODIFIED: July 07, 2017
#
#============================================================================


#------------------- PROBLEM 1:  ------------------------------

f <- function(x, y){
    f <- x*exp(-x^2 - y^2) + (x^2 + y^2)/20
    
    return(f)
}

x <- seq(-2, 2, length = 50)
y <- seq(-2, 2, length = 50)
z <- outer(x, y, f)

postscript(file = 'problem01_3D_01.ps', paper = "a4", horizontal = TRUE)
par(mfrow = c(1, 2))

# Gráfico 3D: 
persp(x, y, z, theta = 35, phi = 25, expand = 0.8, ticktype = "detailed", 
      xlab = "x", ylab="y", zlab = "f(x, y)", col = 'lightblue' , border = 'black', 
      shade = 0.2)


# Curvas de Nível: 
contour(x, y, z, xlab = "x", ylab = "y")

dev.off()
par(mfrow = c(1,1))


# Function to be minimized: 
prob01 <- function(z){
    x <- z[1]
    y <- z[2]
    
    f <- x*exp(-x^2 - y^2) + (x^2 + y^2)/20
    
    return(f)
}

# Gradient vector:
grad01 <- function(z){
    x <- z[1]
    y <- z[2]
    
    fx <- exp(-x^2 - y^2)*(1 - 2*x^2) + x/10
    fy <- y/10 - 2*x*y*exp(-x^2 - y^2)
    
    return(c(fx, fy))
}

# Hessian Matrix:
hessian <- function(z){
    x <- z[1]
    y <- z[2]
    
    fxx <- exp(-x^2 - y^2)*(4*x^3 - 6*x) + 1/10
    fyy <- exp(-x^2 - y^2)*(4*x*y^2 - 2*x) + 1/10
    fxy = fyx = exp(-x^2 - y^2)*(4*(x^2)*y - 2*y)
    
    return(cbind(c(fxx, fyx), c(fxy, fyy)))
}


# Initial Guess:
ig01 <- c(1,1)

#--------------- NELDER-MEAD(Método Simplex) ----------------
flag.prob01.01 <- optim(par = ig01, fn = prob01, 
                        method = "Nelder-Mead")

flag.prob01.01


#----------------------- BFGS -------------------------------
flag.prob01.02 <- optim(par = ig01, fn = prob01,
                gr = grad01, method = "BFGS")


flag.prob01.02


#--------------------- L-BFGS-B -----------------------------
flag.prob01.03 <- optim(par = ig01, fn = prob01, gr = grad01,
                        lower = c(-1,-.5), upper = c(0, 0.5),
                        method = "L-BFGS-B")

flag.prob01.03



#=============== PROBLEM 2: Linear Regression ===============
N = 100

# TRUE PARAMETERS:
BETA0 = 1
BETA1 = 2
SIGMA2 = 2


# Simulation:

# Choosing the Random Number Generator:
RNGkind(kind = "Marsaglia-Multicarry", normal.kind = "Inversion")

# State of the RNG:
#   - The first number codes the kind of RNG.
#     The lowest two decimal digits are in 0:(k-1) where k
#     is the number of available RNGs.
#     The hundreds represent the type of normal generator (starting at 0).

# Generating the explanatory variable x: 
set.seed(2017)
stateRNG1 <- .Random.seed
seed1X <- stateRNG1[2] # This value will be used in other plataforms to generate X
seed2X <- stateRNG1[3] # This value will be used in other plataforms to generate X
X <- runif(N)


# Generating uniform numbers to create the orthogonal error term:
set.seed(1991)
stateRNG2 <- .Random.seed
seed1E <- stateRNG2[2] # This value will be used in other plataforms to generate the error term
seed2E <- stateRNG2[3] # This value will be used in other plataforms to generate the error term
error <- runif(N)


# Simulating tge values of the dependent variable: 
Y <- BETA0 + BETA1*X + sqrt(SIGMA2)*qnorm(error)
data.table(Y = round(Y, 5), X = round(X, 5))
#############



#======================== Linear Regression ========================

# Likelihood function: 
linearreg <- function(theta){
    beta0 <- theta[1]
    beta1 <- theta[2]
    sigma2 <- theta[3]
    
    f <- -N*log(sqrt(sigma2)) - (1/(2*sigma2))*sum((Y - beta0 - beta1*X)^2);
    
    return(f)
}

# Gradient vector:
grad.LR <- function(theta){
    beta0 <- theta[1]
    beta1 <- theta[2]
    sigma2 <- theta[3]
    
    g1  = (1/sigma2)*sum(Y - beta0 - beta1*X)
    g2 =  (1/sigma2)*sum((Y - beta0 - beta1*X)*X)
    g3 =  (1/(2*(sigma2^2)))*sum((Y - beta0 - beta1*X)^2) - (N/(2*sigma2))
    
    return(c(g1, g2, g3))
}


# Initial Guess:
ig.lr <- c(mean(Y), 1, 0.5)


#------------------------- NELDER-MEAD ------------------------------------
flag.OLS.01 <- optim(fn = linearreg, par = ig.lr, 
                     method = "Nelder-Mead", control = list(fnscale = -1))

flag.OLS.01


#------------------------------- BFGS -------------------------------------
flag.OLS.02 <- optim(fn = linearreg, gr = grad.LR,
                     par = ig.lr, method = "BFGS",
                     control = list(fnscale = -1))

flag.OLS.02

#------------------------------ L-BFGS-B -----------------------------------
flag.OLS.03 <- optim(fn = linearreg, gr = grad.LR, 
                     par = ig.lr, method = "L-BFGS-B",
                     lower = c(-Inf, -Inf, 0.00001), control = list(fnscale = -1))

flag.OLS.03

# Checking:
lm(Y~X)



#======================= PROBLEM 3: Logistic Regression =====================
rm(list = ls())

N = 100

# TRUE PARAMETERS:
Beta0 = 0.2
Beta1 = 0.5

# Generating the explanatory variable x: 
set.seed(2306)
stateRNG1 <- .Random.seed
seed1X <- stateRNG1[2] # This value will be used in other plataforms to generate X
seed2X <- stateRNG1[3] # This value will be used in other plataforms to generate X
X <- runif(N)

# Generating standard normal numbers by inversion method:
X <- qnorm(X) # inversion method
X # Normal(0, 1)

# Generating probability of Success:
pi = exp(Beta0 + Beta1*X)/(1 + exp(Beta0 + Beta1*X))
data.table(pi = round(pi, 5), X = round(X, 5))


# Generating uniform numbers to binomial dependent variable:
set.seed(1455)
stateRNG2 <- .Random.seed
seed1E <- stateRNG2[2] # This value will be used in other plataforms to generate Y
seed2E <- stateRNG2[3] # This value will be used in other plataforms to generate Y
u <- runif(N)

Y <- rep(NA, length = N)

for(i in 1:N){
    #print(.Random.seed)
    Y[i] <- qbinom(p = u[i], size = 1, prob = pi[i])
}

# Data generated:
data.table(Y, pi = round(pi, 5), X = round(X, 5))

###################################

# Log-likelihood function:
fun = function(theta){
    beta0 = theta[1]
    beta1 = theta[2]
    pi = exp(beta0 + beta1*X)/(1 + exp(beta0 + beta1*X))
    
    f = sum(Y*log(pi) + (1 - Y)*log(1 - pi))
    
    return(f)
}


# Optimization:

# Initial Guess:
ig.log <- c(0,0)
#---------------------------- NELDER-MEAD -----------------------------------
flag.logit.NM = optim(fn = fun, par = ig.log, method = "Nelder-Mead",
             control = list(fnscale = -1))

flag.logit.NM


#-------------------------------- BFGS --------------------------------------
flag.logit.BFGS = optim(fn = fun, par = ig.log, 
                        method = "BFGS",
                        control = list(fnscale = -1))

flag.logit.BFGS


#------------------------------ L-BFGS-B --------------------------------------
flag.logit.LBFGSB = optim(fn = fun, par = ig.log, 
                          method = "L-BFGS-B",
                          lower = c(0,0),
                          control = list(fnscale = -1))

flag.logit.LBFGSB


# Checking: 
glm(formula = Y~X, family = binomial)































