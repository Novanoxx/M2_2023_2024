#include <PMat.h>

// Les "moteurs" : mise à jour de l’état
// intégrateur Leapfrog

static void update_leapfrog(PMat *M, float h)
{
    M->vit += h*M->frc/M->m;// intégration 1 : vitesse m.F(n) = (V(n+1)-V(n))/h -EXplicite
    M->pos.y += h*M->vit; // intégration 2 : position V(n+1) = (X(n+1)-X(n))/h -IMplicite
    M->frc = 0.; // on vide le buffer de force
}
// intégrateur Euler Explicite
// juste pour l’exemple : méthode très instable -> à éviter
// simple échange des 2 premières lignes par rapport à Leapfrog -> ça change tout
static void update_euler_exp(PMat *M, float h)
{
    M->pos.y += h*M->vit; // intégration 1 : position V(n) = (X(n+1)-X(n-1))/h -EXplicite
    M->vit += h*M->frc/M->m;// intégration 2 : vitesse m.F(n) = (V(n+1)-V(n))/h -EXplicite
    M->frc = 0.; // on vide le buffer de force
}
// mise à jour point fixe : ne fait rien
static void update_fixe(PMat *M, float h)
{
    // position et vitesse restent inchangées
    M->frc = 0.; // on vide le buffer de force (par sécurité)
}

extern void M_builder(PMat *M, int type, float m, G2Xpoint P0, float V0)
{
    M->m = m; // masse
    M->pos = P0; // position initiale
    M->vit = V0; // vitesse initiale
    M->frc = 0; // JAMAIS de force à la création
    switch (type)// choix de la fonction de mise à jour
    {
        case 0 : M->update = update_fixe; break;// Point Fixe
        case 1 : M->update = update_leapfrog; break;// Particule Leapfrog
        case 2 : M->update = update_euler_exp; break;// Particule Euler Exp.
    }
}