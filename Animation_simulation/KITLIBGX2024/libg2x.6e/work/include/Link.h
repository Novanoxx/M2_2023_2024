#include <PMat.h>

typedef struct _lnk_
{
    float k,l,z,s; // paramètres divers
    PMat *M1,*M2; // points M de connexion
    void (*update)(struct _lnk_ *this); //mise à jour : calcul/distrib des forces
} Link;

// Connecteur : branche L entre M1 et M2
void Connect(PMat *M1, Link *L, PMat *M2);
// Constructeurs
void Hook_Spring(Link *L, float k);
void Kinetic_Damper(Link *L, float z);
void Damped_Hook(Link *L, float k, float z);
void Cond_Damped_Hook(Link *L, float k, float z, float s);

// autres constructeurs.... autant qu’on veut