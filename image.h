#ifndef IMAGE_H
#define IMAGE_

#include "SDL.h"
#include <string>

class Image
{
    SDL_Surface *surface_;

    void init(int w, int h) {
        if (surface_)
            SDL_FreeSurface(surface_);

        surface_ = SDL_CreateRGBSurface(SDL_SWSURFACE, w, h, 32, 
                0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff);
    }
    static int CompareInt(const void *a, const void *b) {
        return (*(const int *) a) - (*(const int *) b);
    }
public:
    Image(const std::string &src) : surface_(NULL) {
        if (SDL_Surface *t = SDL_LoadBMP(src.c_str())) {
            init(t->w, t->h);
            SDL_BlitSurface(t, NULL, surface_, NULL);
            SDL_FreeSurface(t);
        }
        else
            throw std::string("unable to load image");
    }
    Image(SDL_Surface *src) : surface_(NULL) {
        init(src->w, src->h);
        SDL_BlitSurface(src, NULL, surface_, NULL);
    }
    Image(int w, int h) : surface_(NULL) {
        init(w, h);
    }
    ~Image() { SDL_FreeSurface(surface_); }

    void BlitTo(SDL_Surface *dest, int x = 0, int y = 0) const {
        SDL_Rect r = {x, y, 0, 0};
        SDL_BlitSurface(surface_, NULL, dest, &r);
    }
    int Height() const { return surface_->h; }
    int Width() const { return surface_->w; }

    uint64_t Compare(const Image &src) const;

    void Clear() { SDL_FillRect(surface_, NULL, 0); }
    void Polygon(const Sint16 * vx, const Sint16 * vy, int n, Uint32 color);
    void HLineAlpha(Sint16 x1, Sint16 x2, Sint16 y, Uint32 r, Uint32 g, Uint32 b, Uint8 alpha); 

    bool Save(const std::string &name) {
        return SDL_SaveBMP(surface_, name.c_str()) == 0;
    }
};

#endif
