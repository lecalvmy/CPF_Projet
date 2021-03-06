﻿#pragma once

#include "Option.hpp"
#include "Model.hpp"
#include "pnl/pnl_random.h"
#include <math.h>

/// \brief Méthode de Monte Carlo pour les calculs
class MonteCarlo
{
public:
	/*! pointeur vers un modèle */
	Model *mod_;
	/*! pointeur sur l'option */
	Option *opt_;
	/*! pointeur sur le générateur de nombres aléatoires */
	PnlRng *rng_;
	/*! pas de h */
	double fdStep_;
	/*! nombre de tirages Monte Carlo */
	int nbSamples_;

	/**
	* Constructeur de la classe
	* param[in] mod : pointeur vers le modèle
	* param[in] opt : pointeur sur l'option
	* param[in] rng : pointeur sur le générateur
	* param[in]  fdStep : pas de différence finie
	* param[in]  nbSamples : nombre de tirages Monte Carlo
	*/
	MonteCarlo(Model *mod, Option *opt, PnlRng *rng, double fdStep, int nbSamples);

	/**
	 * Calcule le prix de l'option à la date 0
	 *
	 * @param[out] prix valeur de l'estimateur Monte Carlo
	 * @param[out] ic largeur de l'intervalle de confiance
	 */
	void price(double &prix, double &ic);

	/**
	 * Calcule le prix de l'option à la date 0 version parallele
	 *
	 * param[in] size : nb threads
 	 * param[in] rank : rang du thread
	 * @param[out] prix valeur de l'estimateur Monte Carlo
	 * @param[out] ic largeur de l'intervalle de confiance
	 */
	void price(double &prix, double &ic, int size, int rank);

	/**
	 * Calcule la somme des payoff et la somme des payoff au carré pour les processus esclaves
	 * et les envoie au processus mettre
	 *
	 * param[in] size : nb threads
	 * param[in] path : la trajectoire générée
	 * param[in] rng : pointeur sur le générateur de nombre aléatoire
 	 * param[in] nbSamples_slave : le nombre de fois qu'il faire la simulation
	 */

	void price_slave(int size, PnlMat *path, PnlRng *rng, int nbSamples_slave);

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

	void price_master(double &prix, double &esp_carre, int size,PnlMat *path, PnlRng *rng, int nbSamples_master);

	/**
	 * Calcule le nombre de tour que le programme doit faire
	 *
	 * @param[out] prix valeur de l'estimateur Monte Carlo
	 * param[in] size : nb threads
	 * param[in] rank : le rank du thread
	 * param[in] precision : L'écart type souhaité
	 * @param[out] nbSamplesNeeded:  Calcule le nombre de tour que le programme doit faire
	 */

	void price(double &prix, int size, int rank, double precision, double &nbSamplesNeeded, double &ictheorique);



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
	void price(const PnlMat *past, double t, double &prix, double &ic);

	/**
	 * Calcule le delta de l'option à la date t
	 *
	 * @param[in] past contient la trajectoire du sous-jacent
	 * jusqu'à l'instant t
	 * @param[in] t date à laquelle le calcul est fait
	 * @param[out] delta contient le vecteur de delta
	 * @param[in] conf_delta contient le vecteur d'intervalle de confiance sur le calcul du delta
	 */
	void delta(const PnlMat *past, double t, PnlVect *delta, PnlVect *conf_delta);

	/* destructeur pour la classe MonteCarlo */
	~MonteCarlo();

};
