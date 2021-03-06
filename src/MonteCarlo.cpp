﻿#include "MonteCarlo.hpp"
#include <time.h>
#include "mpi.h"
using namespace std;

/**
* Constructeur de la classe
* param[in] mod : pointeur vers le modèle
* param[in] opt : pointeur sur l'option
* param[in] rng : pointeur sur le générateur
* param[in] fdStep : pas de différence finie
* param[in] nbSamples : nombre de tirages Monte Carlo
*/
MonteCarlo::MonteCarlo(Model *mod, Option *opt, PnlRng *rng, double fdStep, int nbSamples) {
	mod_ = mod;
	opt_ = opt;
	rng_ = rng;
	fdStep_ = fdStep;
	nbSamples_ = nbSamples;
}

/**
	* Calcule le prix de l'option à la date 0
	* @param[out] prix valeur de l'estimateur Monte Carlo
	* @param[out] ic largeur de l'intervalle de confiance
*/

void MonteCarlo::price(double &prix, double &ic) {

	double payoff;
	prix =  0;
	double esp_carre = 0;

	pnl_rng_init(rng_, PNL_RNG_MERSENNE);
    pnl_rng_sseed(rng_, time(NULL));

	PnlMat *path = pnl_mat_create(opt_->nbTimeSteps_ + 1, mod_->size_);
	pnl_mat_set_row(path, mod_->spot_, 0);
	for (int j = 0; j < nbSamples_; ++j) {
	mod_->asset(path, opt_->T_, opt_->nbTimeSteps_, rng_);
		payoff = opt_->payoff(path);
		prix += payoff;
		esp_carre += (payoff * payoff);
	}
	double estimateur_carre = exp(-2 * mod_->r_*opt_->T_)*(esp_carre / nbSamples_ - (prix/nbSamples_)*(prix/nbSamples_));
	prix *= exp(-mod_->r_*opt_->T_) / nbSamples_;
	ic = 1.96 * sqrt(estimateur_carre / nbSamples_);

	pnl_mat_free(&path);
}

/**
 * Calcule le prix de l'option à la date 0 version parallele
 *
 * param[in] size : nb threads
 * param[in] rank : rang du thread
 * @param[out] prix valeur de l'estimateur Monte Carlo
 * @param[out] ic largeur de l'intervalle de confiance
 */

void MonteCarlo::price(double &prix, double &ic, int size, int rank){
	prix = 0;

	PnlRng *rng = pnl_rng_dcmt_create_id(rank, 1234);
	pnl_rng_sseed(rng, time(NULL));

	PnlMat *path = pnl_mat_create(opt_->nbTimeSteps_ + 1, mod_->size_);
	pnl_mat_set_row(path, mod_->spot_, 0);

	int nbSamples_master = nbSamples_ / size + nbSamples_ % size;
	int nbSamples_slave = nbSamples_ / size;

	double esp_carre = 0;

	if(rank != 0)
    {
		price_slave(size, path, rng, nbSamples_slave);
    }
    else if (rank == 0)
    {
		price_master(prix, esp_carre, size, path, rng, nbSamples_master);
		double estimateur_carre = exp(-2 * mod_->r_*opt_->T_)*(esp_carre / nbSamples_ - (prix/nbSamples_)*(prix/nbSamples_));
		prix *= exp(-mod_->r_*opt_->T_) / nbSamples_;
		ic = 1.96 * sqrt(estimateur_carre / nbSamples_);
    }

	pnl_rng_free(&rng);
	pnl_mat_free(&path);
}

/**
 * Calcule la somme des payoff et la somme des payoff au carré pour les processus esclaves
 * et les envoie au processus mettre
 *
 * param[in] size : nb threads
 * param[in] path : la trajectoire générée
 * param[in] rng : pointeur sur le générateur de nombre aléatoire
 * param[in] nbSamples_slave : le nombre de fois qu'il faire la simulation
 */

void MonteCarlo::price_slave(int size, PnlMat *path, PnlRng *rng, int nbSamples_slave){

	double payoff, prix_th, esp_carre_th;

	for (int j = 0; j < nbSamples_slave; ++j) {
		mod_->asset(path, opt_->T_, opt_->nbTimeSteps_, rng);
		payoff = opt_->payoff(path);
		prix_th += payoff;
		esp_carre_th += (payoff * payoff);
	}

	MPI_Send(&prix_th,1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
	MPI_Send(&esp_carre_th,1,MPI_DOUBLE,0,2,MPI_COMM_WORLD);
}

/**
 * Calcule la somme des payoff et la somme des payoff au carré de tous les processus
 *
 * @param[out] prix il s'agit de la somme des payoffs
 * @param[out] esp_carre il s'agit de la somme des payoffs au carré
 * param[in] size : nb threads
 * param[in] path : la trajectoire générée
 * param[in] rng : pointeur sur le générateur de nombre aléatoire
 * param[in] nbSamples_master : le nombre de fois qu'il faire la simulation
 */


void MonteCarlo::price_master(double &prix, double &esp_carre, int size,PnlMat *path, PnlRng *rng, int nbSamples_master){

	double prix_th, esp_carre_th, payoff ;

	for (int j = 0; j < nbSamples_master; ++j) {
		mod_->asset(path, opt_->T_, opt_->nbTimeSteps_, rng);
		payoff = opt_->payoff(path);
		prix += payoff;
		esp_carre += (payoff * payoff);
	}

	for(int i=1; i<size; ++i){
		MPI_Recv(&prix_th,1,MPI_DOUBLE,MPI_ANY_SOURCE, 1,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		prix += prix_th;
		MPI_Recv(&esp_carre_th,1,MPI_DOUBLE,MPI_ANY_SOURCE, 2,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		esp_carre += esp_carre_th;
	}
}

/**
 * Calcule le nombre de tour que le programme doit faire
 *
 * @param[out] prix valeur de l'estimateur Monte Carlo
 * param[in] size : nb threads
 * param[in] rank : le rank du thread
 * param[in] precision : L'écart type souhaité
 * @param[out] nbSamplesNeeded:  Calcule le nombre de tour que le programme doit faire
 */

void MonteCarlo::price(double &prix, int size, int rank, double precision, double &nbSamplesNeeded, double &ictheorique){

	PnlRng *rng = pnl_rng_dcmt_create_id(rank, 1234);
	pnl_rng_sseed(rng, time(NULL));

	PnlMat *path = pnl_mat_create(opt_->nbTimeSteps_ + 1, mod_->size_);
	pnl_mat_set_row(path, mod_->spot_, 0);

	nbSamplesNeeded = 0;
	double sumPayoff = 0;
	double esp_carre = 0;
	double estimateur_carre;
	//double ic;
	do {
		if (rank == 0)
		{
			price_master(sumPayoff, esp_carre, size, path, rng, 1);
			nbSamplesNeeded += size;
			double meanPrice = sumPayoff/nbSamplesNeeded;
			estimateur_carre = exp(-2 * mod_->r_*opt_->T_)*(esp_carre / nbSamplesNeeded - meanPrice*meanPrice);
			prix = exp(-mod_->r_*opt_->T_) * meanPrice;
			ictheorique = 1.96 * sqrt(estimateur_carre / nbSamplesNeeded);
			for(int i = 1; i < size; ++i){
				MPI_Send(&ictheorique, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD);
			}
		}
		else if(rank != 0)
		{
			price_slave(size, path, rng, 1);
			MPI_Recv(&ictheorique, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

	  MPI_Barrier(MPI_COMM_WORLD);
	} while(ictheorique > precision);
	pnl_rng_free(&rng);
	pnl_mat_free(&path);
}



/**
 * Calcule le prix de l'option à la date t
 *
 * @param[in]  past contient la trajectoire du sous-jacent
 * jusqu'à l'instant t
 * @param[in] t date à laquelle le calcul est fait
 * @param[out] prix contient le prix
 * @param[out] ic contient la largeur de l'intervalle
 * de confiance sur le calcul du prix
 */
void MonteCarlo::price(const PnlMat *past, double t, double &prix, double &ic) {

	double payoff;
	prix = 0;
	double esp_carre = 0;

	PnlMat *path = pnl_mat_create(opt_->nbTimeSteps_ + 1, opt_->size_);
	pnl_mat_set_subblock(path, past, 0, 0);

	for (int i = 0; i < nbSamples_; ++i) {
		mod_->asset(path, t, opt_->T_, opt_->nbTimeSteps_, rng_, past);
		payoff = opt_->payoff(path);
		prix += payoff;
		esp_carre += pow(payoff, 2);
	}

	double estimateur_carre = exp(-2 * mod_->r_*(opt_->T_ - t))*(esp_carre / nbSamples_ - pow(prix / nbSamples_, 2));
	prix *= exp(-mod_->r_*(opt_->T_ - t)) / nbSamples_;
	ic = 1.96 * sqrt(estimateur_carre / nbSamples_);

	pnl_mat_free(&path);
}

/**
 * Calcule le delta de l'option à la date t
 *
 * @param[in] past contient la trajectoire du sous-jacent
 * jusqu'à l'instant t
 * @param[in] t date à laquelle le calcul est fait
 * @param[out] delta contient le vecteur de delta
 * @param[in] conf_delta contient le vecteur d'intervalle de confiance sur le calcul du delta
 */
void MonteCarlo::delta(const PnlMat *past, double t, PnlVect *delta, PnlVect *conf_delta) {

	double sum;
	double sum2;
	double timestep = opt_->T_ / (opt_->nbTimeSteps_);
	double coefficient;
	double prix;
	double payoff_increment;
	double payoff_decrement;
	double standard_dev;

	PnlMat *path = pnl_mat_create(opt_->nbTimeSteps_ + 1, mod_->size_);
	PnlMat *increment_path = pnl_mat_create(opt_->nbTimeSteps_ + 1, mod_->size_);
	PnlMat *decrement_path = pnl_mat_create(opt_->nbTimeSteps_ + 1, mod_->size_);

	pnl_mat_set_subblock(path, past, 0, 0);

	for (int d = 0; d < mod_->size_; d++) {

		sum = 0;
		sum2 = 0;
		prix = MGET(past, past->m - 1, d);
		coefficient = exp(-mod_->r_ * (opt_->T_ - t)) / (2 * fdStep_ * prix);

		for (int i = 0; i < nbSamples_; i++) {

			mod_->asset(path, t, opt_->T_, opt_->nbTimeSteps_, rng_, past);
			mod_->shiftAsset(increment_path, path, d, fdStep_, t, timestep);
			mod_->shiftAsset(decrement_path, path, d, -fdStep_, t, timestep);

			payoff_increment = opt_->payoff(increment_path);
			payoff_decrement = opt_->payoff(decrement_path);

			sum += payoff_increment - payoff_decrement;
			sum2 += pow(payoff_increment - payoff_decrement, 2);

		}

		pnl_vect_set(delta, d, coefficient * sum / nbSamples_);
		standard_dev = coefficient * sqrt(sum2 / nbSamples_ - pow(sum / nbSamples_, 2));
		pnl_vect_set(conf_delta, d, standard_dev / sqrt(nbSamples_));

	}

	pnl_mat_free(&path);
	pnl_mat_free(&increment_path);
	pnl_mat_free(&decrement_path);
}

MonteCarlo::~MonteCarlo()
{
	delete mod_;
	delete opt_;
	pnl_rng_free(&rng_);
}
