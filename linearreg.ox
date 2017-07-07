# include <oxstd.oxh>
# include <oxprob.oxh>
# import <maximize>


// Declaring Global variables:
static decl s_vy;
static decl s_vx;
static const decl N = 100;

//------------------------------ LINEAR REGRESSION --------------------------------

linearreg01(const vPar, const adVal, const avGrad, const amHess){
	decl b0 = vPar[0];
	decl b1 = vPar[1];
	decl sigma2 = vPar[2];

	// Likelihood function to be
	adVal[0] = -N*log(sqrt(sigma2)) - (1/(2*sigma2))*sumsqrc(s_vy - (b0 + b1*s_vx));
						
	if(avGrad){
		(avGrad[0])[0] = (1/sigma2)*sumc(s_vy - (b0 + b1*s_vx));
		(avGrad[0])[1] = (1/sigma2)*sumc((s_vy - (b0 + b1*s_vx)).*s_vx);
		(avGrad[0])[2] = (1/(2*(sigma2^2)))*sumsqrc(s_vy - (b0 + b1*s_vx)) - (N/(2*sigma2));
	}
							
	if(isnan(adVal[0]) || isdotinf(adVal[0])){
		return 0;
		} else{
			return 1;
		}
}

//---------------------------------------------------------------------------------

linearreg02(const vPar, const adVal, const avGrad, const amHess){
	decl b0 = vPar[0];
	decl b1 = vPar[1];
	decl sigma2 = vPar[2];

	// Likelihood function to be maximized:
	adVal[0] = -N*log(sqrt(sigma2)) - (1/(2*sigma2))*sumsqrc(s_vy - (b0 + b1*s_vx));

	// Gradient Vector:
	if(avGrad){
		(avGrad[0])[0] = (1/sigma2)*sumc(s_vy - (b0 + b1*s_vx));
		(avGrad[0])[1] = (1/sigma2)*sumc((s_vy - (b0 + b1*s_vx)).*s_vx);
		(avGrad[0])[2] = (1/(2*(sigma2^2)))*sumsqrc(s_vy - (b0 + b1*s_vx)) - (N/(2*sigma2));
	}
						
	// Hessian Matrix (Newton-Raphson):
	if(amHess){
		(amHess[0])[0][0] = -N/sigma2;
		(amHess[0])[0][1] = (-1/sigma2)*sumc(s_vx);
		(amHess[0])[0][2] = (-1/(sigma2^2))*sumc(s_vy - (b0 + b1*s_vx));
		(amHess[0])[1][0] =  (-1/sigma2)*sumc(s_vx);						
		(amHess[0])[1][1] = (-1/sigma2)*sumsqrc(s_vx);
		(amHess[0])[1][2] = (-1/(sigma2^2))*sumc((s_vy - (b0 +  b1*s_vx)) .* s_vx);
		(amHess[0])[2][0] = (-1/(sigma2^2))*sumc(s_vy - (b0 + b1*s_vx));
		(amHess[0])[2][1] = (-1/(sigma2^2))*sumc((s_vy - (b0 +  b1*s_vx)) .* s_vx);
		(amHess[0])[2][2] = N/(2*(sigma2^2)) - (1/(sigma2^3))*sumc(s_vy - b0 - b1*s_vx);
	}

	if(isnan(adVal[0]) || isdotinf(adVal[0])){
		return 0;
		} else{
			return 1;
		}
}

//---------------------------------------------------------------------------------

linearreg03(const vPar, const adVal, const avGrad, const amHess){
	decl b0 = vPar[0];
	decl b1 = vPar[1];
	decl sigma2 = vPar[2];

	// Likelihood function to be maximized:
	adVal[0] = -N*log(sqrt(sigma2)) - (1/(2*sigma2))*sumsqrc(s_vy - (b0 + b1*s_vx));

	// Gradient Vector:
	if(avGrad){
		(avGrad[0])[0] = (1/sigma2)*sumc(s_vy - (b0 + b1*s_vx));
		(avGrad[0])[1] = (1/sigma2)*sumc((s_vy - (b0 + b1*s_vx)).*s_vx);
		(avGrad[0])[2] = (1/(2*(sigma2^2)))*sumsqrc(s_vy - (b0 + b1*s_vx)) - (N/(2*sigma2));
	}
						
	// Steepest Descent:
	if(amHess){
		(amHess[0])[0][0] = (amHess[0])[1][1] = (amHess[0])[2][2] = -1;
		(amHess[0])[0][1] = (amHess[0])[0][2] = (amHess[0])[1][0] = (amHess[0])[1][2] = (amHess[0])[2][0] = (amHess[0])[2][1] = 0;
	}


	if(isnan(adVal[0]) || isdotinf(adVal[0])){
		return 0;
		} else{
			return 1;
		}
}

//---------------------------------------------------------------------------------



main(){
	decl flag_LR_BFGS, vPar_LR_BFGS, dVal_LR_BFGS;
	decl flag_LR_Newton, vPar_LR_Newton, dVal_LR_Newton;
	decl flag_LR_SD, vPar_LR_SD, dVal_LR_SD;


	// True Parameters: 
	decl SIGMA2= 2;
	decl BETA = <1; 2>;

	// Setting the random number generator: 
	ranseed("GM");

	// Generating the explanatory variable s_vx: 
	ranseed({-874407620, 1370210573}); 
	s_vx = ranu(N, 1);

	// Generating standard uniform distributed numbers: 
	ranseed({-71132134, 417223379}); 
	decl error = ranu(N, 1);

	// Generating the dependent variable s_vy: 
	s_vy = BETA[0] + BETA[1]*s_vx + sqrt(SIGMA2)*quann(error); // Inversion method
	print("%c", {"Y", "X"}, s_vy~s_vx);



	//------------------------------- STEEPEST DESCENT ------------------------------

	// Initial value:
	vPar_LR_SD = meanc(s_vy)|1|0.5;

	flag_LR_SD = MaxNewton(linearreg03, &vPar_LR_SD, &dVal_LR_SD, 0, FALSE);

	print("\n");
	print("\nRESULTS FOR THE LINEAR REGRESSION:\n");
	print("\nCONVERGENCE: ", "         ", MaxConvergenceMsg(flag_LR_SD));
	print("\nMETHOD: ", "              ", "Steepest Descent");
	print("\nHESSIAN: ", "             ", "Analytical");
	print("\nSAMPLE SIZE: ", "% 12d", N);
	print("\n", "LIKELIHOOD VALUE: ", "%10.3f", double(dVal_LR_SD));
	print("\n\nTRUE PARAMETERS: ", "%r", {"beta0", "beta1", "sigma2"}, "%14.3f", BETA|SIGMA2);
	print("\nML Estimators: ", "%r", {"beta0^", "beta1^", "sigma2^"}, "%14.3f", vPar_LR_SD);
	print("\n\n\n");


	//-------------------------------- NEWTON-RAPHSON -------------------------------

	// Initial value:
	vPar_LR_Newton = meanc(s_vy)|1|0.5;

	flag_LR_Newton = MaxNewton(linearreg02, &vPar_LR_Newton, &dVal_LR_Newton, 0, FALSE);

	print("\n");
	print("\nRESULTS FOR THE LINEAR REGRESSION:\n");
	print("\nCONVERGENCE: ", "         ", MaxConvergenceMsg(flag_LR_Newton));
	print("\nMETHOD: ", "              ", "Newton-Raphson");
	print("\nHESSIAN: ", "             ", "Analytical");
	print("\nSAMPLE SIZE: ", "% 12d", N);
	print("\n", "LIKELIHOOD VALUE: ", "%10.3f", double(dVal_LR_Newton));
	print("\n\nTRUE PARAMETERS: ", "%r", {"beta0", "beta1", "sigma2"}, "%14.3f", BETA|SIGMA2);
	print("\nML Estimators: ", "%r", {"beta0^", "beta1^", "sigma2^"}, "%14.3f", vPar_LR_Newton);
	print("\n\n\n");


	//--------------------------------- BFGS ----------------------------------------

	// Initial value:
	vPar_LR_BFGS = meanc(s_vy)|1|0.5;

	// Optimization:
	flag_LR_BFGS = MaxBFGS(linearreg01, &vPar_LR_BFGS, &dVal_LR_BFGS, 0, FALSE);
	

	print("\n");
	print("\nRESULTS FOR THE LINEAR REGRESSION:\n");
	print("\nCONVERGENCE: ", "         ", MaxConvergenceMsg(flag_LR_BFGS));
	print("\nMETHOD: ", "              ", "BGFS");
	print("\nGRADIENT: ", "            ", "Analytical");
	print("\nSAMPLE SIZE: ", "% 12d", N);
	print("\n", "LIKELIHOOD VALUE: ", "%10.3f", double(dVal_LR_BFGS));
	print("\n\nTRUE PARAMETERS: ", "%r", {"beta0", "beta1", "sigma2"}, "%14.3f", BETA|SIGMA2);
	print("\nML Estimators: ", "%r", {"beta0^", "beta1^", "sigma2^"}, "%14.3f", vPar_LR_BFGS);
	print("\n\n\n");

}




