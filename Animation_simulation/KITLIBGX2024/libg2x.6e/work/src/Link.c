#include <Link.h>

// Connecteur : branche L entre M1 et M2
extern void Connect(PMat *M1, Link *L, PMat *M2)
{
    L->M1 = M1;
    L->M2 = M2;
    // longueur "à vide" (ressort et assimilés) :
    // L->l = M2->pos.y - M1->pos.y;
    L->l = g2x_Dist(M1->pos, M2->pos);
}

// Les "moteurs" : calcul et distribution des forces
// Ressort linéaire
static void update_Hook(Link *L)
{
    float d = g2x_Dist(L->M1->pos, L->M2->pos); // distance courante ∣∣−−−→M1M2∣∣
    G2Xvector u = g2x_Vector2p(L->M1->pos, L->M2->pos);
    u.x = (d != 0.) ? -u.x/d : 1.; // direction M1 → M2
    u.y = (d != 0.) ? -u.y/d : 1.; // direction M1 → M2 M1 → M2
    G2Xvector F;
    F.x = -L->k*(d-L->l)*u.x; // force de rappel
    F.y = -L->k*(d-L->l)*u.y; // force de rappel

    L->M1->frc = g2x_SubVect(L->M1->frc, F); // distribution sur M1
    L->M2->frc = g2x_AddVect(L->M2->frc, F); // distribution sur M2
}
// Frein cinétique linéaire
static void update_Damper(Link *L)
{
    G2Xvector F;
    F.x = -L->z*(L->M2->vit.x - L->M1->vit.x); // force de freinage
    F.y = -L->z*(L->M2->vit.y - L->M1->vit.y); // force de freinage

    L->M1->frc = g2x_SubVect(L->M1->frc, F); // distribution sur M1
    L->M2->frc = g2x_AddVect(L->M2->frc, F); // distribution sur M2
}
// Combinaison des 2 (montage en parallèle)
static void update_Damped_Hook(Link *L)
{
    float d = g2x_Dist(L->M1->pos, L->M2->pos); // distance courante ∣∣−−−→M1M2∣∣
    G2Xvector u = g2x_Vector2p(L->M1->pos, L->M2->pos);
    u.x = (d != 0.) ? -u.x/d : 1.; // direction M1 → M2
    u.y = (d != 0.) ? -u.y/d : 1.; // direction M1 → M2 M1 → M2
    G2Xvector F;
    F.x = -L->k*(d - L->l)*u.x -L->z*(L->M2->vit.x-L->M1->vit.x); // force combinées
    F.y = -L->k*(d - L->l)*u.y -L->z*(L->M2->vit.y-L->M1->vit.y); // force combinées
    
    L->M1->frc = g2x_AddVect(L->M1->frc, F); // distribution sur M1
    L->M2->frc = g2x_SubVect(L->M2->frc, F); // distribution sur M2
}
// Liaison Ressort-Frein conditionnelle
// avec L->s=1. : simple "choc" visco-élastique
// avec L->s>1. : lien "inter-particule" - légère adhérence pour d ∈ [l0, s ∗ l0]
static void update_Cond_Damped_Hook(Link *L)
{
    float d = g2x_Dist(L->M1->pos, L->M2->pos); // distance courante ∣∣−−−→M1M2∣∣
    if (d>L->s * L->l) return;
    G2Xvector u = g2x_Vector2p(L->M1->pos, L->M2->pos);
    u.x = (d != 0.) ? -u.x/d : 1.; // direction M1 → M2
    u.y = (d != 0.) ? -u.y/d : 1.; // direction M1 → M2 M1 → M2
    G2Xvector F;
    F.x = -L->k*(d - L->l)*u.x -L->z*(L->M2->vit.x-L->M1->vit.x); // force combinées
    F.y = -L->k*(d - L->l)*u.y -L->z*(L->M2->vit.y-L->M1->vit.y); // force combinées
    
    L->M1->frc = g2x_AddVect(L->M1->frc, F); // distribution sur M1
    L->M2->frc = g2x_SubVect(L->M2->frc, F); // distribution sur M2
}

// Les constructeurs : choix de la nature de la liaison
// Ressort linéaire
extern void Hook_Spring(Link *L, float k)
{
    L->k = k;
    L->update = update_Hook;
}
// Frein cinétique linéaire
extern void Kinetic_Damper(Link *L, float z)
{
    L->z = z;
    L->update = update_Damper;
}
// Combinaison des 2 (montage en parallèle)
extern void Damped_Hook(Link *L, float k, float z)
{
    L->k = k;
    L->z = z;
    L->update = update_Damped_Hook;
}
// Liaison Ressort-Frein conditionnelle
extern void Cond_Damped_Hook(Link *L, float k, float z, float s)
{
    L->k = k;
    L->z = z;
    L->s = s;
    L->update = update_Cond_Damped_Hook;
}