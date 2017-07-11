#==========================================================================
# PROGRAM: optimization.R
#
# LANGUAGE*: R
#
# DESCIPTION: This file contains the necessary routine to perform 
#             optimization using the optim() and maxLik() from the
#             packages stats and maxLik, repespectively, in the 
#             following examples:
#               - minimize the function: x*exp(-x^2 - y^2) + (x^2 + y^2)/20
#               - maximize the log-likelihood function for a linear regression 
#               - maximize the log-likelihood function for a logit regression
#
# DATE: June 21, 2017
#
# LAST MODIFIED: July 11, 2017
#==========================================================================

library(data.table)


#============================== PROBLEM 1: ================================

f <- function(x, y){
    f <- x*exp(-x^2 - y^2) + (x^2 + y^2)/20
    
    return(f)
}

x <- seq(-2, 2, length = 50)
y <- seq(-2, 2, length = 50)
z <- outer(x, y, f)

library(plot3D)

postscript(file = 'problem01_3D_01.ps', paper = "a4", horizontal = TRUE)

persp3D(x = x, y = y, z = z,  xlab = "x", ylab = "y", zlab = "f(x, y)",
        colkey = list(length = 0.2, width = 0.4, shift = 0.15,
                      cex.axis = 0.8, cex.clab = 0.85), lighting = TRUE, lphi = 90,
        plot = T,contour = T, ticktype = "detailed", theta = 35, phi = 25)

#title(expression(f(x,y) == x*e^(-x^2 - x^2) + frac((x^2 + y^2),20)))
dev.off()



# Function to be minimized: 
fun01 <- function(z){
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
hessian01 <- function(z){
    x <- z[1]
    y <- z[2]
    
    fxx <- exp(-x^2 - y^2)*(4*x^3 - 6*x) + 1/10
    fyy <- exp(-x^2 - y^2)*(4*x*y^2 - 2*x) + 1/10
    fxy = fyx = exp(-x^2 - y^2)*(4*(x^2)*y - 2*y)
    
    return(cbind(c(fxx, fyx), c(fxy, fyy)))
}


# Initial Guess:
ig01 <- c(-1,0)

#-------------------- NELDER-MEAD(Método Simplex) ------------------------
flag.prob01.01 <- optim(par = ig01, fn = fun01, 
                        method = "Nelder-Mead")

flag.prob01.01
list(convergence = flag.prob01.01$convergence,
     value_function = round(flag.prob01.01$value, 6),
     estimates = round(flag.prob01.01$par, 6))


#----------------------------- BFGS --------------------------------------
flag.prob01.02 <- optim(par = ig01, fn = fun01,
                gr = grad01, method = "BFGS")

flag.prob01.02
list(convergence = flag.prob01.02$convergence,
     value_function = round(flag.prob01.02$value, 6),
     estimates = round(flag.prob01.02$par, 6))


#--------------------------- L-BFGS-B ------------------------------------
flag.prob01.03 <- optim(par = ig01, fn = fun01, gr = grad01,
                        lower = c(-1,-1), upper = c(1,1),
                        method = "L-BFGS-B")

flag.prob01.03
list(convergence = flag.prob01.03$convergence,
     value_function = round(flag.prob01.03$value, 6),
     estimates = round(flag.prob01.03$par, 6))





#=================== PROBLEM 2: Linear Regression ========================
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



#======================== Linear Regression ==============================

# log-Likelihood function for linear regression:
LR <- function(theta){
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


# Hessian matrix:
hessian.LR <- function(theta){
    beta0 <- theta[1]
    beta1 <- theta[2]
    sigma2 <- theta[3]
    
    h11 = -N/sigma2
    h12 = (-1/sigma2)*sum(X)
    h13 = (-1/(sigma2^2))*sum(Y - (beta0 + beta1*X))
    h21 = h12
    h22 = (-1/sigma2)*sum(X^2)
    h23 = (-1/(sigma2^2))*sum((Y - (beta0 +  beta1*X))*X)
    h31 = h13
    h32 = h23
    h33 = N/(2*(sigma2^2)) - (1/(sigma2^3))*sum((Y - beta0 - beta1*X)^2)

    return(cbind(c(h11, h21, h31), c(h12, h22, h32), c(h13, h23, h33)))
}



# Initial Guess:
ig.lr <- c(mean(Y), 1, 0.5)


#------------------------- NELDER-MEAD ------------------------------------
flag.OLS.01 <- optim(fn = LR, par = ig.lr, 
                     method = "Nelder-Mead", 
                     control = list(fnscale = -1))

flag.OLS.01
list(convergence = flag.OLS.01$convergence,
     value_funciton = round(flag.OLS.01$value, 6),
     estimates = round(flag.OLS.01$par, 6))



#------------------------------- BFGS -------------------------------------
flag.OLS.02 <- optim(fn = LR, gr = grad.LR,
                     par = ig.lr, method = "BFGS",
                     control = list(fnscale = -1))

flag.OLS.02
list(convergence = flag.OLS.02$convergence,
     value_funciton = round(flag.OLS.02$value, 6),
     estimates = round(flag.OLS.02$par, 6))

#------------------------------ L-BFGS-B ----------------------------------
flag.OLS.03 <- optim(fn = LR, gr = grad.LR, 
                     par = ig.lr, method = "L-BFGS-B",
                     lower = c(-Inf, -Inf, 0.00001), 
                     control = list(fnscale = -1))

flag.OLS.03
list(convergence = flag.OLS.03$convergence,
     value_funciton = round(flag.OLS.03$value, 6),
     estimates = round(flag.OLS.03$par, 6))

# Checking the results:
# lm(Y~X)



#======================= PROBLEM 3: Logistic Regression ===================
N = 100
# TRUE PARAMETERS:
Beta0 = 0.2
Beta1 = 0.5

# Generating the explanatory variable x: 
set.seed(2306)
stateRNG1 <- .Random.seed
seed1X <- stateRNG1[2] # This value will be used in other plataforms to generate X
seed2X <- stateRNG1[3] # This value will be used in other plataforms to generate X
XX <- runif(N)

# Generating standard normal numbers by inversion method:
XX <- qnorm(XX) # inversion method
XX # Normal(0, 1)

# Generating probability of Success:
pi = exp(Beta0 + Beta1*XX)/(1 + exp(Beta0 + Beta1*XX))
data.table(pi = round(pi, 5), X = round(XX, 5))


# Generating uniform numbers to create the binary  dependent variable YY:
set.seed(1455)
stateRNG2 <- .Random.seed
seed1E <- stateRNG2[2] # This value will be used in other plataforms to generate Y
seed2E <- stateRNG2[3] # This value will be used in other plataforms to generate Y
u <- runif(N)

YY <- rep(NA, length = N)

for(i in 1:N){
    #print(.Random.seed)
    YY[i] <- qbinom(p = u[i], size = 1, prob = pi[i])
}

# Data generated:
data.table(Y = YY, pi = round(pi, 5), X = round(XX, 5))

###################################

# Log-likelihood function for logistic regresion:
log.R <- function(theta){
    beta0 = theta[1]
    beta1 = theta[2]
    pi = exp(beta0 + beta1*XX)/(1 + exp(beta0 + beta1*XX))
    
    f = sum(YY*log(pi) + (1 - YY)*log(1 - pi))
    
    return(f)
}


# Gradient vector: 
grad.log.R <-  function(theta){
    beta0 = theta[1]
    beta1 = theta[2]
    pi = exp(beta0 + beta1*XX)/(1 + exp(beta0 + beta1*XX))
    
    g1 = sum(YY - pi)
    g2 = sum(XX * (YY - pi))
    
    return(c(g1, g2))
        
}


# Hessian Matrix: 
Hessian.log.R <- function(theta){
    beta0 = theta[1]
    beta1 = theta[2]
    pi = exp(beta0 + beta1*XX)/(1 + exp(beta0 + beta1*XX))
    
    h11 = (-1.0)*sum(exp(-beta0 - beta1*XX)*(pi^2))
    h12 = (-1.0)*sum(XX*exp(-beta0 - beta1*XX)*(pi^2))
    h21 = h12
    h22 = (-1.0)*sum((XX^2)*exp(-beta0 - beta1*XX)*(pi^2))
    
    return(cbind(c(h11, h21), c(h12, h22)))
}




# Optimization:
ig.log <- c(0,0)
#---------------------------- NELDER-MEAD ---------------------------------
flag.logit.NM = optim(fn = log.R, par = ig.log, method = "Nelder-Mead",
             control = list(fnscale = -1))

flag.logit.NM
list(convergence = flag.logit.NM$convergence,
     value_functon = round(flag.logit.NM$value, 6),
     estimates = round(flag.logit.NM$par, 6))


#-------------------------------- BFGS ------------------------------------
flag.logit.BFGS = optim(fn = log.R, par = ig.log, gr = grad.log.R,
                        method = "BFGS",
                        control = list(fnscale = -1))

flag.logit.BFGS
list(convergence = flag.logit.BFGS$convergence,
     value_functon = round(flag.logit.BFGS$value, 6),
     estimates = round(flag.logit.BFGS$par, 6))


#------------------------------ L-BFGS-B ----------------------------------
flag.logit.LBFGSB = optim(fn = log.R, par = ig.log, gr = grad.log.R,
                          method = "L-BFGS-B",
                          lower = c(-1,-1),
                          control = list(fnscale = -1))

flag.logit.LBFGSB
list(convergence = flag.logit.LBFGSB$convergence,
     value_functon = round(flag.logit.LBFGSB$value, 6),
     estimates = round(flag.logit.LBFGSB$par, 6))

# Checking the results: 
#summary(glm(formula = YY~XX, family = binomial))




#========================== USING THE PACKAGE maxLik ======================
library(maxLik)


#------------------------- PROBLEM 01 ------------------------------------

# By default, maxLik performs maximization. In order to minimize a
# function, we have to transform a minimization problem into a maximization
# one by multiplying the target-function by -1: 

fun01.1 <- function(z){
            x <- z[1]
            y <- z[2]
    
            f <- (-1)*(x*exp(-x^2 - y^2) + (x^2 + y^2)/20)
    
        return(f)
}

grad01.1 <- function(z){
        x <- z[1]
        y <- z[2]
        
        fx <- (-1)*(exp(-x^2 - y^2)*(1 - 2*x^2) + x/10)
        fy <- (-1)*(y/10 - 2*x*y*exp(-x^2 - y^2))
        
        return(c(fx, fy))
}


hessian01.1 <- function(z){
    x <- z[1]
    y <- z[2]
    
    fxx <- (-1)*(exp(-x^2 - y^2)*(4*x^3 - 6*x) + 1/10)
    fyy <- (-1)*(exp(-x^2 - y^2)*(4*x*y^2 - 2*x) + 1/10)
    fxy = fyx = (-1)*(exp(-x^2 - y^2)*(4*(x^2)*y - 2*y))
    
    return(cbind(c(fxx, fyx), c(fxy, fyy)))
}




#--------------------------------- BFGS -----------------------------------
mL.prob01.BFGS <- maxLik(logLik = fun01.1, grad = grad01.1,
                         start = ig01, method = "BFGS")

# Results: 
list(minimum = (-1)*round(mL.prob01.BFGS$maximum, 6),
     estimate = round(mL.prob01.BFGS$estimate, 6))




#---------------------------- Newton-Raphson ------------------------------
mL.prob01.Newton <- maxLik(logLik = fun01.1, grad = grad01.1, hess = hessian01.1,
                         start = ig01, method = "NR")
mL.prob01.Newton

# Results: 
list(minimum = (-1)*round(mL.prob01.Newton$maximum, 6),
     estimate = round(mL.prob01.Newton$estimate, 6))



#---------------------------------- BHHH ----------------------------------
# Using BHHH algorithm, the gradient function must return a matrix where
# rows corresponds to the gradient vectors for individual observations
# and the columns to the individual parameters. 

grad01.1.BHHH <- function(z){
    x <- z[1]
    y <- z[2]
    
    fx <- (-1)*(exp(-x^2 - y^2)*(1 - 2*x^2) + x/10)
    fy <- (-1)*(y/10 - 2*x*y*exp(-x^2 - y^2))
    g <- c(fx, fy)
    gg <- rbind(g, g)
    
    return(gg)
}


mL.prob01.BHHH <- maxBHHH(fn = fun01.1, grad =  grad01.1.BHHH, 
                        start = ig01)

# Results: 
list(Minimum = round(-1*mL.prob01.BHHH$maximum, 6),
           estimates = round(mL.prob01.BHHH$estimate, 6))




#======================= PROBLEM 02: LINEAR REGRESSION ====================


#-------------------------------- BFGS ------------------------------------
mL.LR.BFGS <- maxLik(logLik = LR, grad = grad.LR,
                     method = "BFGS", start = ig.lr)

# Results:
list(log_likelihood_value = round(mL.LR.BFGS$maximum, 6),
     estimates = round(mL.LR.BFGS$estimate, 6))


#---------------------------- NEWTON-RAPHSON ------------------------------
mL.LR.Newton <- maxLik(logLik = LR, grad = grad.LR, hess = hessian.LR,
                       method = "NR", start = ig.lr)
# Results: 
list(log_likelihood = round(mL.LR.Newton$maximum, 6),
     estimates = round(mL.LR.Newton$estimate, 6))

#--------------------------------- BHHH -----------------------------------

grad.LR.BHHH <- function(theta){
    beta0 <- theta[1]
    beta1 <- theta[2]
    sigma2 <- theta[3]
    
    g1 <- rep(0, N)
    g2 <- rep(0, N)
    g3 <- rep(0, N)
    
    for(i in 1:N){
        g1[i]  = (1/sigma2)*(Y[i] - beta0 - beta1*X[i])
        g2[i] =  (1/sigma2)*((Y[i] - beta0 - beta1*X[i])*X[i])
        g3[i] =  (1/(2*(sigma2^2)))*((Y[i] - beta0 - beta1*X[i])^2) - (1/(2*sigma2)) 
    }
    
    gg <- cbind(g1, g2, g3)
    return(gg)
}


mL.LR.BHHH <- maxBHHH(fn = LR,grad = grad.LR.BHHH, start = ig.lr)

# Results: 
list(Minimum = round(mL.LR.BHHH$maximum, 6),
     estimates = round(mL.LR.BHHH$estimate, 6))




#==================== PROBLEM 03: LOGISTIC REGRESSION =====================

#----------------------------- BFGS ---------------------------------------
mL.logit.BFGS <- maxLik(logLik = log.R, grad = grad.log.R, method = "BFGS",
                        start = ig.log)
mL.logit.BFGS

# Results: 
list(log_likelihood = round(mL.logit.BFGS$maximum, 6),
     estiamtes = round(mL.logit.BFGS$estimate, 6))



#-------------------------- Newton-Raphson --------------------------------
mL.logit.Newton <- maxLik(logLik = log.R, grad = grad.log.R, hess = Hessian.log.R,
                          method = "NR", start = ig.log)
mL.logit.Newton

# Results: 
list(log_likelihood = round(mL.logit.Newton$maximum, 6),
     estiamtes = round(mL.logit.Newton$estimate, 6))



#------------------------------ BHHH --------------------------------------

grad.log.R.BHHH <- function(theta){
    beta0 = theta[1]
    beta1 = theta[2]
    pi = exp(beta0 + beta1*XX)/(1 + exp(beta0 + beta1*XX))
    
    g1 <- rep(0, N)
    g2 <- rep(0, N)
    
    for(i in 1:N){
        g1[i] <- (YY[i] - pi[i])
        g2[i] <- (YY[i] - pi[i])*XX[i]
    }
    
    gg <- cbind(g1, g2)
    
    return(gg)
}

mL.logit.BHHH <- maxBHHH(fn = log.R,grad =  grad.log.R.BHHH,
                         start = ig.log)

# Results: 
list(log_likelihood = round(mL.logit.BHHH$maximum, 6),
     estiamtes = round(mL.logit.BHHH$estimate, 6))

##########################################################################
