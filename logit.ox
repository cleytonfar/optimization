/**********************************************************************************
* FILE: logit.ox
*
* LANGUAGE: Ox
*
* DESCRIPTION: 	This file contains programming routine to perform non-linear
*	 	optimization. Specifically, this routine is applied to 
*		optimize the log-likelihood function for a logistic regression.
*              	To this end, I employed three algorithms to achieve the optimal
*              	conditions of the target function. 
*
* AUTHOR: Cleyton Farias
*
* DATE: June 14, 2017
*
* LAST MODIFIED: July 11, 2017
*
**********************************************************************************/

# include <oxstd.oxh>
# include <oxprob.h>
# import <maximize>

// Declaring the Global Variables:
static decl s_vY;
static decl s_vX;
static const decl N = 100;


//----------------------------- LOGIT REGRESSION  ---------------------------------

logit01(const vPar, const adVal, const avGrad, const amHess){
	decl beta0 = vPar[0];
	decl beta1 = vPar[1];
	decl pi = exp(beta0 + beta1*s_vX) ./ (1 + exp(beta0 + beta1*s_vX)); 

	// log-likelihood:
	adVal[0] = sumc(s_vY .* log(pi) + (1 - s_vY) .* log(1 - pi) );

	// Gradient vector:
	if(avGrad){
		(avGrad[0])[0] = sumc(s_vY - pi);
		(avGrad[0])[1] = sumc( s_vX .* (s_vY - pi));
		}

	// STEEPEST DESCENT:
	if(amHess){
		(amHess[0])[0][0] = -1;
		(amHess[0])[0][1] = (amHess[0])[1][0] = 0;
		(amHess[0])[1][1] = -1;
		}
							
						
	if(isnan(adVal[0]) || isdotinf(adVal[0])){
		return 0;
		} else{
		return 1;
	}
}


//---------------------------------------------------------------------------------

logit02(const vPar, const adVal, const avGrad, const amHess){
	decl beta0 = vPar[0];
	decl beta1 = vPar[1];
	decl pi = exp(beta0 + beta1*s_vX) ./ (1 + exp(beta0 + beta1*s_vX)); 

	// log-likelihood:
	adVal[0] = sumc(s_vY .* log(pi) + (1 - s_vY) .* log(1 - pi) );

	// Gradient vector:
	if(avGrad){
		(avGrad[0])[0] = sumc(s_vY - pi);
		(avGrad[0])[1] = sumc( s_vX .* (s_vY - pi));
	}

	// NEWTON-RAPHSON:
	if(amHess){
		(amHess[0])[0][0] = (-1.0)*sumc( exp(-beta0 - beta1*s_vX) .* (pi.^2) ) ;
		(amHess[0])[0][1] = (amHess[0])[1][0] = (-1.0)*sumc( s_vX .* exp(-beta0 - beta1*s_vX) .* (pi.^2) );
		(amHess[0])[1][1] = (-1.0)*sumc( (s_vX.^2) .* exp(-beta0 - beta1*s_vX) .* (pi.^2)  );
		}
							
						
	if(isnan(adVal[0]) || isdotinf(adVal[0])){
		return 0;
		} else{
			return 1;
		}
}


//---------------------------------------------------------------------------------

logit03(const vPar, const adVal, const avGrad, const amHess){
	decl beta0 = vPar[0];
	decl beta1 = vPar[1];
	decl pi = exp(beta0 + beta1*s_vX) ./ (1 + exp(beta0 + beta1*s_vX)); 

	// log-likelihood:
	adVal[0] = sumc(s_vY .* log(pi) + (1 - s_vY) .* log(1 - pi) );

	// Gradient vector:
	if(avGrad){
		(avGrad[0])[0] = sumc(s_vY - pi);
		(avGrad[0])[1] = sumc( s_vX .* (s_vY - pi));
	}

								
	if(isnan(adVal[0]) || isdotinf(adVal[0])){
		return 0;
		} else{
			return 1;
		}
}

//---------------------------------------------------------------------------------
								


main(){

	decl flag_log_BFGS, vPar_log_BFGS, dVal_log_BFGS;
	decl flag_log_Newton, vPar_log_Newton, dVal_log_Newton;
	decl flag_log_SD, vPar_log_SD, dVal_log_SD;
	decl pi, u, i;

	// TRUE PARAMETERS:
	decl Beta0 = 0.2;
	decl Beta1 = 0.5;
	
	// Setting the random number generator: 
	ranseed("GM");

	// Generating the explanatory variable s_vX: 
	ranseed({934294641, -1087063170}); 
	s_vX = ranu(N, 1);
	s_vX = quann(s_vX);

	// Generating the probabilities: 
	pi = exp(Beta0 + Beta1*s_vX) ./ (1 + exp(Beta0 + Beta1*s_vX));

	// Generating the dependent variable by inversion method: 
	s_vY = zeros(N, 1);

	// Setting seed to generate standard uniform numbers:
	ranseed({-691168222, 263569723});
	u = ranu(N, 1);

	for(i = 0; i < N; i++){
		//print(ranseed(0), "\n");
		s_vY[i] = quanbinomial(u[i], 1, pi[i]);
	}

	print("%c", {"Y", "pi", "X"}, s_vY~pi~s_vX, "\n");



	//----------------------------- STEEPEST DESCENT --------------------------------

	// Initial guess:
	vPar_log_SD = <0;0>;

	// Optimization:
	flag_log_SD = MaxNewton(logit01, &vPar_log_SD, &dVal_log_SD, 0, FALSE);

	print("\n");
	print("RESULTS FOR THE LOGISTIC REGRESSION:\n");
	print("\nCONVERGENCE: ", "         ", MaxConvergenceMsg(flag_log_SD));
	print("\nMETHOD: ", "              ", "Steepest Descent");
	print("\nHESSIAN: ", "             ", "Analytical");
	print("\nSAMPLE SIZE: ", "%12d", N);
	print("\nLog likelihood: ", "% 15.6f", double(dVal_log_SD));
	print("\n\nTRUE PARAMETERS: ", "%r", {"BETA0", "BETA1"}, "%21.3f", Beta0|Beta1);
	print("\nML Estimators: ", "%r", {"beta0^", "beta1^"}, "%17.6f",vPar_log_SD);
	print("\n\n");
	


	//------------------------------- NEWTON-RAPHSON --------------------------------

	// Initial guess:
	vPar_log_Newton = <0;0>;

	// Optimization:
	flag_log_Newton = MaxNewton(logit02, &vPar_log_Newton, &dVal_log_Newton, 0, FALSE);

	print("\n");
	print("RESULTS FOR THE LOGISTIC REGRESSION:\n");
	print("\nCONVERGENCE: ", "         ", MaxConvergenceMsg(flag_log_Newton));
	print("\nMETHOD: ", "              ", "Newton-Raphson");
	print("\nHessian: ", "             ", "Analytical");
	print("\nSAMPLE SIZE: ", "%12d", N);
	print("\nLog likelihood: ", "% 15.6f", double(dVal_log_Newton));
	print("\n\nTRUE PARAMETERS: ", "%r", {"BETA0", "BETA1"}, "%21.3f", Beta0|Beta1);
	print("\nML Estimators: ", "%r", {"beta0^", "beta1^"}, "%17.6f",vPar_log_Newton);
	print("\n\n");


	//--------------------------------- BFGS ----------------------------------------

	// Initial guess:
	vPar_log_BFGS = <0;0>;

	// Optimization:
	flag_log_BFGS = MaxBFGS(logit03, &vPar_log_BFGS, &dVal_log_BFGS, 0, FALSE);

	print("\n");
	print("RESULTS FOR THE LOGISTIC REGRESSION:\n");
	print("\nCONVERGENCE: ", "         ", MaxConvergenceMsg(flag_log_BFGS));
	print("\nMETHOD: ", "              ", "BGFS");
	print("\nGRADIENT: ", "            ", "Analytical");
	print("\nSAMPLE SIZE: ", "%12d", N);
	print("\nLog likelihood: ", "% 15.6f", double(dVal_log_BFGS));
	print("\n\nTRUE PARAMETERS: ", "%r", {"BETA0", "BETA1"}, "%21.3f", Beta0|Beta1);
	print("\nML Estimators: ", "%r", {"beta0^", "beta1^"}, "%17.6f",vPar_log_BFGS);
	print("\n\n");
	
}
