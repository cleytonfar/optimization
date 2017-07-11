/**********************************************************************************
*
*
*
*
*
**********************************************************************************/


# include <oxstd.oxh>
# import <maximize>


fun01(const vPar, const adVal, const avGrad, const amHess){
	decl x = vPar[0];
	decl y = vPar[1];

	// Function to be Minimized:
	adVal[0] = -x*exp(-x^2 - y^2) - (x^2 + y^2)/20;

	// Gradient Vector: 
	if(avGrad){
		(avGrad[0])[0] = exp(-x^2 - y^2)*(2*x^2 - 1) - x/10;
		(avGrad[0])[1] = exp(-x^2 - y^2)*(2*x*y) - y/10;
	}
	
	if( isnan(adVal[0]) || isdotinf(adVal[0]) ){
		return 0;
		} else{
			return 1;
		}
}



fun02(const vPar, const adVal, const avGrad, const amHess){
		decl x = vPar[0];
		decl y = vPar[1];

		// Function to be Minimized:
		adVal[0] = -x*exp(-x^2 - y^2) - (x^2 + y^2)/20;

		// Gradient Vector: 
		if(avGrad){
			(avGrad[0])[0] = exp(-x^2 - y^2)*(2*x^2 - 1) - x/10;
			(avGrad[0])[1] = exp(-x^2 - y^2)*(2*x*y) - y/10;
		}
	

		// Hessian Matrix:
		if(amHess){
			(amHess[0])[0][0] = exp(-x^2 - y^2)*(-4*x^3 + 6*x) - 1/10;
			(amHess[0])[0][1] = (amHess[0])[1][0] = exp(-x^2 - y^2)*(2*y - 4*x^2*y);
			(amHess[0])[1][1] = (-1.0)*(exp(-x^2 - y^2)*(-2*x + 4*x*(y^2)) + 1/10);
		}
					
							
		if( isnan(adVal[0]) || isdotinf(adVal[0]) ){
			return 0;
			} else{
				return 1;
			}
}





fun03(const vPar, const adVal, const avGrad, const amHess){
	decl x = vPar[0];
	decl y = vPar[1];

	// Function to be Minimized:
	adVal[0] = -x*exp(-x^2 - y^2) - (x^2 + y^2)/20;

	// Gradient Vector: 
	if(avGrad){
		(avGrad[0])[0] = exp(-x^2 - y^2)*(2*x^2 - 1) - x/10;
		(avGrad[0])[1] = exp(-x^2 - y^2)*(2*x*y) - y/10;
	}
	
	// STEEPEST DECENT:
	if(amHess){
		(amHess[0])[0][0] = -1;
		(amHess[0])[0][1] = (amHess[0])[1][0] = 0;
		(amHess[0])[1][1] = -1;
	}

	if( isnan(adVal[0]) || isdotinf(adVal[0]) ){
		return 0;
		} else{
			return 1;
	}
}



/*
fun04(const vPar, const adVal, const avGrad, const amHess){
	decl x = vPar[0];
	decl y = vPar[1];

	// Function to be Minimized:
	adVal[0] = -x*exp(-x^2 - y^2) - (x^2 + y^2)/20;

	// Gradient Vector: 
	if(avGrad){
		(avGrad[0])[0] = exp(-x^2 - y^2)*(2*x^2 - 1) - x/10;
		(avGrad[0])[1] = exp(-x^2 - y^2)*(2*x*y) - y/10;
	}
	
	// BHHH:
	decl gg = (-1.0)*((avGrad[0])*(avGrad[0])');
	if(amHess){
		(amHess[0])[0][0] = gg[0][0];
		(amHess[0])[0][1] = gg[0][1];
		(amHess[0])[1][0] = gg[1][0];
		(amHess[0])[1][1] = gg[1][1];
	}

	if( isnan(adVal[0]) || isdotinf(adVal[0]) ){
		return 0;
		} else{
			return 1;
		}
}
*/





main(){

	decl dFunBFGS, vParBFGS, flagBFGS;
	decl dFunNewton, vParNewton, flagNewton;
	decl dFunSD, vParSD, flagSD;
	decl dFunBHHH, vParBHHH, flagBHHH;


	// BFGS with Analytical Gradient:
	vParBFGS = <-1;0>;
	flagBFGS = MaxBFGS(fun02, &vParBFGS, &dFunBFGS, FALSE, FALSE);

	print("\nResults using BFGS Method:\n");
	print("\nCONVERGENCE:", "       ", MaxConvergenceMsg(flagBFGS), "\n");
	println("METHOD: ", "           ", "BFGS");
	println("GRADIENT:", "          ", "Analytical", "\n");
	println("VALUE FUNCTION: ", "% 10.6f", -1*dFunBFGS);
	println("PARAMETERS:", "%r", {"X", "Y"}, "% 18.6f", vParBFGS);



	// Newton-Raphson with Analytical Hessian:
	vParNewton = <-1;0>;
	flagNewton = MaxNewton(fun02, &vParNewton, &dFunNewton, FALSE, FALSE);

	print("\n");
	print("Results using Newton-Raphson Method:\n");
	print("\nCONVERGENCE:", "       ", MaxConvergenceMsg(flagNewton), "\n");
	println("METHOD: ", "           ", "Newton-Raphson");
	println("HESSIAN:", "           ", "Analytical", "\n");
	println("VALUE FUNCTION: ", "% 10.6f", -1*dFunNewton);
	println("PARAMETERS:", "%r", {"X", "Y"}, "% 18.6f", vParNewton);


	// Steepest Descent:
	vParSD = <-1;0>;
	//MaxControlEps(0.0000001, 0.0000001);
	flagSD = MaxNewton(fun03, &vParSD, &dFunSD, FALSE, FALSE);

	print(GetMaxControlEps());
	print("\n");
	print("Results using Steepest Descent Method:\n");
	print("\nCONVERGENCE:", "       ", MaxConvergenceMsg(flagSD), "\n");
	println("METHOD: ", "           ", "Steepest Descent");
	println("HESSIAN:", "          ", "-Identity", "\n");
	println("VALUE FUNCTION: ", "% 10.6f", -1*dFunSD);
	println("PARAMETERS:", "%r", {"X", "Y"}, "% 18.6f", vParSD);

/*
	// BHHH:
	vParBHHH = <-1;0>;
	flagBHHH = MaxNewton(fun04, &vParBHHH, &dFunBHHH, FALSE, FALSE);

	print("\n");
	print("Results using BHHH Method:\n");
	print("\nCONVERGENCE:", "       ", MaxConvergenceMsg(flagBHHH), "\n");
	println("METHOD: ", "           ", "BHHH");
	println("HESSIAN:", "          ", "-Grad*Grad'", "\n");
	println("VALUE FUNCTION: ", "% 10.6f", -1*dFunBHHH);
	println("PARAMETERS:", "%r", {"X", "Y"}, "% 18.6f", vParBHHH);
*/
	
}












					
