/**********************************************************************************
* FILE: uncoptim.c
*
* LANGUAGE: C
*
* DESCRIPTION: This file contains programming routine to perform non-linear
*	             optimization. Specifically, I employed three algorithms to 
*              achieve the optimal  conditions of the target function: 
*                   f(x, y) = x*exp(-x*x - y*y) + (x*x + y*y)/10,
*              which has no analytical solution.
*
* AUTHOR: Cleyton Farias
*
* DATE: July 08, 2017
*
* LAST MODIFIED: July 11, 2017
*
**********************************************************************************/

# include <stdio.h>
# include <stdlib.h>
# include "gsl/gsl_multimin.h"


// Function to be minimized: 
double fun(const gsl_vector *v, void *params){

	double x, y;
	x = gsl_vector_get(v, 0);
	y = gsl_vector_get(v, 1);
	
	double *p = (double *) params;
	
	// Function to be Minimized:
	return  (x*exp(-x*x - y*y) + (x*x + y*y)/p[0]);
	
}


// Gradient vector: 
void Gradfun(const gsl_vector *v, void *params, gsl_vector *df){

	double x, y;
	x = gsl_vector_get(v, 0);
	y = gsl_vector_get(v, 1);

	double *p = (double *) params;

	gsl_vector_set(df, 0,  exp(-x*x - y*y)*(1 - 2*x*x) + x/10);
	gsl_vector_set(df, 1,  y/10 - 2*x*y*exp(-x*x - y*y));

}


// Compute both fun and Gradfun together (more efficient):
void my_fdf(const gsl_vector *v, void *params, double *f, gsl_vector *df){
	
	*f = fun(v, params);
	Gradfun(v, params, df);
	
}

//=================================================================================


int  main(void){

	int  iter = 0;
	int flagCG, flagBFGS, flagSD;

	const gsl_multimin_fdfminimizer_type *T1, *T2, *T3;
	gsl_multimin_fdfminimizer *s1, *s2, *s3;
		
	double p[] = {20.0};

	gsl_multimin_function_fdf my_func;
	my_func.n = 2;
	my_func.f = &fun;
	my_func.df = &Gradfun;
	my_func.fdf = &my_fdf;
	my_func.params = p;

	// Initial Guess:
	gsl_vector *v;
	v = gsl_vector_alloc(2);
	gsl_vector_set(v, 0, -1.0);
	gsl_vector_set(v, 1, 0.0);

	
	// Chossing the algoritms to be used:	
	T1 = gsl_multimin_fdfminimizer_vector_bfgs2;
	T2 = gsl_multimin_fdfminimizer_steepest_descent;
	T3 = gsl_multimin_fdfminimizer_conjugate_fr;

	// Setting the minimizers:
	s1 = gsl_multimin_fdfminimizer_alloc(T1, 2);
	s2 = gsl_multimin_fdfminimizer_alloc(T2, 2);
	s3 = gsl_multimin_fdfminimizer_alloc(T3, 2);


	//--------------------------- BFGS ------------------------------
	gsl_multimin_fdfminimizer_set(s1, &my_func, v, 0.01, 1e-4);
	
	printf("\nResults using BFGS algorithm:\n");
	
	
	do{
		iter++;

		flagBFGS = gsl_multimin_fdfminimizer_iterate(s1);
		
		if(flagBFGS){
			break;
		}

		flagBFGS = gsl_multimin_test_gradient(s1->gradient, 1e-6);
		
		if(flagBFGS == GSL_SUCCESS){
			printf("\nMinimum found at: \n");
			printf("    i \t  x\ty\tF(x, y)\n");
			printf("%5d %.6f %.6f % 10.6f\n", iter,
					 gsl_vector_get(s1-> x, 0),
					 gsl_vector_get(s1-> x, 1),
					 s1-> f);
		}
		
	}while(flagBFGS == GSL_CONTINUE && iter < 100);

	gsl_multimin_fdfminimizer_free(s1);

	
	//--------------------- STEEPEST DESCENT ------------------------
	gsl_multimin_fdfminimizer_set(s2, &my_func, v, 0.01, 1e-4);
	
	printf("\n\n\nResults using STEEPEST DESCENT algorithm:\n");
	iter = 0;
	
	do{
		iter++;

		flagSD = gsl_multimin_fdfminimizer_iterate(s2);
		
		if(flagSD){
			break;
		}
		
		flagSD = gsl_multimin_test_gradient(s2->gradient, 1e-5);

		if(flagSD == GSL_SUCCESS){
			printf("\nMinimum found at: \n");
			printf("   i \t   x\t y\tF(x, y)\n");
			printf("% 5d  %.6f  %.6f % 10.6f\n", iter,
					 gsl_vector_get(s2-> x, 0),
					 gsl_vector_get(s2-> x, 1),
					 s2-> f);
		}

			
	}while(flagSD == GSL_CONTINUE && iter < 200);

	gsl_multimin_fdfminimizer_free(s2);


	
	//-------------- Fletcher-Reeves Conjugate algorithm --------------
	gsl_multimin_fdfminimizer_set(s3, &my_func, v, 0.01, 1e-4);
	
	printf("\n\n\nResults using FLETCHER-REEVES CONJUGATE algorithm:\n");
	iter = 0;
	
	do{
		iter++;

		flagCG = gsl_multimin_fdfminimizer_iterate(s3);
		
		if(flagCG){
			break;
		}
		
		flagCG = gsl_multimin_test_gradient(s3->gradient, 1e-5);

		if(flagCG == GSL_SUCCESS){
			printf("\nMinimum found at: \n");
			printf("    i \t   x\t y\tF(x, y)\n");
			printf("% 5d  %.6f  %.6f % 10.6f\n", iter,
					 gsl_vector_get(s3-> x, 0),
					 gsl_vector_get(s3-> x, 1),
					 s3-> f);
		}
		
	}while(flagCG == GSL_CONTINUE && iter < 200);

	gsl_multimin_fdfminimizer_free(s3);


	
	gsl_vector_free(v);		


	return 0;
}



	



	






					
