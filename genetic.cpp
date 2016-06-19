/*
 * Simple Genetic Programming demo
 *
 * Based on C# implementation from Roger Alsing (http://rogeralsing.com)
 *
 * This is about 10/20 times faster than .net implementation and produce similar
 * results.
 *
 * Written by Gabriele Greco (gabrielegreco@gmail.com)
 *
 */

#include "point.h"
#include <assert.h>
#include <list>
#include <iostream>
#include <sstream>
#include "image.h"

#include <omp.h>

#include <sys/time.h>
#include <time.h>

#define RangeRand(x) (rand() % (x))
#define IntervalRand(a,b) ((rand() % (b-a + 1)) + a)
#define DoMutate(x) ((rand() % Settings::x) == 1)
#define R255 (rand() % 256)

namespace Settings
{
    int ScreenWidth;
    int ScreenHeight;

    int MaxPoints = 10;
    int MinPoints = 3;
    int MinPolygons = 3;
    int MaxPolygons = 150;
    int AlphaMin = 20;
    int AlphaMax = 120;
    int PointMidMovement = 20;
    int PointSmallMovement = 3;

    // mutations
    int ComponentMutation = 1500;
    int AddPolyMutation = 700;
    int MovePolyMutation = 700;
    int DelPolyMutation = 1500;
    int AddPointMutation = 1500;
    int DelPointMutation = 1500;
    int AlphaMutation = 1500;
    int PointMinMutation = 1500;
    int PointMidMutation = 1500;
    int PointMaxMutation = 1500;

    // flags
};

typedef Point<int16_t> Point16;

class DnaPoint : public Point16
{
public:    
    DnaPoint() {}
    DnaPoint(int16_t x, int16_t y) : Point16(x, y) {}

    bool Mutate() {
        bool dirty = false;

        if (DoMutate(PointMaxMutation)) {
            X(RangeRand(Settings::ScreenWidth));
            Y(RangeRand(Settings::ScreenHeight));
            dirty = true;
        }
        if (DoMutate(PointMidMutation)) {
            DnaPoint p(RangeRand(Settings::PointMidMovement) * 2 - Settings::PointMidMovement, 
                       RangeRand(Settings::PointMidMovement) * 2 - Settings::PointMidMovement);
            *this += p;
            dirty = true;
        }
        if (DoMutate(PointMinMutation)) {
            DnaPoint p(RangeRand(Settings::PointSmallMovement) * 2 - Settings::PointSmallMovement, 
                       RangeRand(Settings::PointSmallMovement) * 2 - Settings::PointSmallMovement);
            *this += p;
            dirty = true;
        }
        if (X() < 0) X(0);
        if (Y() < 0) Y(0);
        if (X() >= Settings::ScreenWidth)  X(Settings::ScreenWidth - 1);
        if (Y() >= Settings::ScreenHeight) Y(Settings::ScreenHeight - 1);

        return dirty;
    }
};

class DnaPolygon : public Serializable
{
    typedef std::list< DnaPoint > PointVec;
    typedef PointVec::iterator PointIt;
    PointVec vertex_;
    uint32_t color_;
public:
    DnaPolygon()  {
        vertex_.push_back(DnaPoint(RangeRand(Settings::ScreenWidth), 
                                   RangeRand(Settings::ScreenHeight)));
        vertex_.push_back(DnaPoint(RangeRand(Settings::ScreenWidth), 
                                   RangeRand(Settings::ScreenHeight)));
        vertex_.push_back(DnaPoint(RangeRand(Settings::ScreenWidth), 
                                   RangeRand(Settings::ScreenHeight)));
        Color(R255, R255, R255);
        Alpha(IntervalRand(Settings::AlphaMin, Settings::AlphaMax));
    }

    void Render(Image &dest) const {
        int points = vertex_.size(); 
        Sint16 x[points];
        Sint16 y[points];
        int i = 0;

        for ( PointVec::const_iterator it = vertex_.begin(); 
              it != vertex_.end(); ++it, ++i) {
            x[i] = it->X();
            y[i] = it->Y();
        }

        dest.Polygon(x, y, points, color_);
    }
    bool Color(int r, int g, int b) { 
        color_ &= 0xff;
        color_ |= ((uint32_t)r << 24 | (uint32_t)g << 16 | (uint32_t)b << 8);
        return true;
    }
    bool RandomRed() {
        color_ &= 0xffffff;
        color_ |= (R255 << 24);
        return true;
    }
    bool RandomGreen() {
        color_ &= 0xff00ffff;
        color_ |= (R255 << 16);
        return true;
    }
    bool RandomBlue() {
        color_ &= 0xffff00ff;
        color_ |= (R255 << 8);
        return true;
    }
    bool Alpha(int v) { 
        color_ &= 0xffffff00; 
        color_ |= (v&0xff); 
        return true;
    }
    bool AddPoint() {
        if (vertex_.size() == Settings::MaxPoints)
            return false;

        int idx = RangeRand(vertex_.size());
        PointIt it = vertex_.begin();
        std::advance(it, idx);
        vertex_.insert(it, DnaPoint(RangeRand(Settings::ScreenWidth), 
                                    RangeRand(Settings::ScreenHeight)));

        return true;
    }
    bool DelPoint() {
        if (vertex_.size() == Settings::MinPoints)
            return false;

        int idx = RangeRand(vertex_.size());
        PointIt it = vertex_.begin();
        std::advance(it, idx);
        vertex_.erase(it);

        return true;
    }
    bool Mutate() {
        bool dirty = false;

        if (DoMutate(AddPointMutation))
            dirty |= AddPoint();
        if (DoMutate(DelPointMutation))
            dirty |= DelPoint();
        if (DoMutate(AlphaMutation))
            dirty |= Alpha(IntervalRand(Settings::AlphaMin, Settings::AlphaMax));

        if (DoMutate(ComponentMutation)) dirty |= RandomRed();
        if (DoMutate(ComponentMutation)) dirty |= RandomGreen();            
        if (DoMutate(ComponentMutation)) dirty |= RandomBlue();

        for (PointIt it = vertex_.begin(); it != vertex_.end(); ++it)
            dirty |= it->Mutate();

        return dirty;
    }

    void serialize(Serializer &s) {
        s & vertex_;
        s & color_;
    }
};

class DnaDrawing : public Serializable
{
    typedef std::list<DnaPolygon> PolyList;
    typedef PolyList::iterator PolyIt;
    PolyList poly_;
public:    
    int Poly() const { return poly_.size(); }
    DnaDrawing() {
        for (int i = 0; i < Settings::MinPolygons; ++i)
            poly_.push_back(DnaPolygon());
    }
    bool DelPoly() {
        if (poly_.size() == Settings::MinPolygons)
            return false;

        int idx = RangeRand(poly_.size());
        PolyIt it = poly_.begin();
        std::advance(it, idx);
        poly_.erase(it);
        return true;
    }
    bool AddPoly() {
        if (poly_.size() == Settings::MaxPolygons)
            return false;

        int idx = RangeRand(poly_.size());
        PolyIt it = poly_.begin();
        std::advance(it, idx);
        poly_.insert(it, DnaPolygon());
        return true;
    }

    bool MovePoly() {
        int origpos = RangeRand(poly_.size());
        int newpos  = RangeRand(poly_.size());

        if (newpos == origpos)
            return false;

        PolyIt it = poly_.begin();
        std::advance(it, origpos);
        DnaPolygon p(*it);
        poly_.erase(it);
        it = poly_.begin();
        std::advance(it, newpos);
        poly_.insert(it, p);

        return true;
    }

    bool Mutate() {
        bool dirty = false;

        if (DoMutate(AddPolyMutation))
            dirty |= AddPoly(); 
        if (DoMutate(DelPolyMutation))
            dirty |= DelPoly();
        if (DoMutate(MovePolyMutation))
            dirty |= MovePoly();

        for (PolyIt it = poly_.begin(); it != poly_.end(); ++it)
            dirty |= it->Mutate();

        return dirty;
    }

    void Render(Image &dest) const {
        dest.Clear();

        for (PolyList::const_iterator it = poly_.begin(); it != poly_.end(); ++it)
            it->Render(dest);
    }

    void serialize(Serializer &s) {
        s & poly_;
    }
};

class GeneticTest : public Serializable
{
    const Image &ref_;
    Image dest_;
    DnaDrawing drawing_;
    uint64_t error_;
    uint32_t generation_;
public:    
    GeneticTest(const Image &ref) : 
        ref_(ref), dest_(ref.Width(), ref.Height()),
        error_(99999999999999999LL),
        generation_(1) {
    }

    void Save(const std::string &name) {
        dest_.Save(name);
    }
    bool Step() {
        DnaDrawing d = drawing_;

        while(!d.Mutate());

        d.Render(dest_);
        generation_++;
        int64_t newErrorLevel = dest_.Compare(ref_);

        if (newErrorLevel < error_) {
            error_ = newErrorLevel;
            drawing_ = d;

            return true;
        }

        return false;
    }
    void Display(SDL_Surface *video) {
        drawing_.Render(dest_);
        dest_.BlitTo(video);
        SDL_UpdateRect(video, 0, 0, dest_.Width(), dest_.Height());
    }
    void serialize(Serializer &s) {
        s & drawing_;
        s & error_;
        s & generation_;
    }
    uint64_t Error() const { return error_; }
    uint32_t Generation() const { return generation_; }
    int Poly() const { return drawing_.Poly(); }
};

int main(int argc, char *argv[])
{
    int width = 640;
    int height = 480;
    Image *source = NULL;
    std::string basename = "noname";
    std::string workfilename;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Unable to initialize SDL!\n";
        return -1;
    }

    atexit(SDL_Quit);

    if (argc >= 2) {
        try {
            source = new Image(argv[1]);
        }
        catch (const std::string &err) {
            std::cerr << "Error in image creation: " << err << "\n";
            return -1;
        }

        width = source->Width();
        height = source->Height();
        basename = argv[1];

        if (argc == 3)
            workfilename = argv[2];
    }
    else {
        std::cerr << "**** IMAGE NOT SPECIFIED ****!\n";
        return -1;
    }

    SDL_Surface *video = SDL_SetVideoMode(width * 2, height, 32, SDL_SWSURFACE);

    source->BlitTo(video, width, 0);
    SDL_UpdateRect(video, width, 0, width, height);

    if (!video) {
        std::cerr << "Unable to open window!\n";
        return -1;
    }

    Settings::ScreenHeight = source->Height();
    Settings::ScreenWidth = source->Width();

    uint64_t counter = 0;
    time_t start = ::time(NULL);
    bool running = true;
    time_t last = start;
    ::srand(start);

    /* Set time measure */
    struct timeval t_start, t_stop;
    unsigned long t_elapsed;
    gettimeofday(&t_start, NULL);

    GeneticTest gen(*source);

    if (!workfilename.empty()) {
        std::cerr << "Loading progresses from <" << workfilename << ">...\n";
        if (FILE *f = ::fopen(workfilename.c_str(), "rb")) {
            SerialLoader l(f);
            l & gen;
            ::fclose(f);
            gen.Display(video);
        }
    }

    while (running) {
        if (gen.Step()) {
            time_t now = time(NULL);

            if (now > last) {
                gen.Display(video);
                last = now;
            }
        }

        counter++;

        if ((counter % 100) == 0) {
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                if (e.type == SDL_QUIT )
                    running = false;
                else if (e.type == SDL_KEYDOWN) {
                    if (e.key.keysym.sym == SDLK_q)
                        running = false;
                    else if (e.key.keysym.sym == SDLK_s) {
                        if (FILE *f = fopen((basename + ".idx").c_str(), "wb")) {
                            SerialSaver s(f);
                            s & gen;
                            fclose(f);
                            std::cerr << "Saved progresses to <" << basename << ".idx>.\n";
                            SDL_WM_SetCaption(("Saved to " + basename + ".idx").c_str(), NULL);
                        }
                    }
                    else if (e.key.keysym.sym == SDLK_b) {
                        gen.Save(basename + ".out.bmp");
                        std::cerr << "Saved BMP to <" << basename << ".out.bmp>.\n";
                        SDL_WM_SetCaption(("Saved BMP to " + basename + ".out.bmp").c_str(), NULL);
                    }
                    counter = 1;
                }
            }
        }
        if ((counter % 2000) == 0) {
            time_t now = time(NULL);
            now -= start;

            /* Get elapsed time */
            gettimeofday(&t_stop, NULL);
            t_elapsed = 1000000 * (t_stop.tv_sec - t_start.tv_sec);
            t_elapsed += t_stop.tv_usec - t_start.tv_usec;

            if (now > 0) {
                std::cerr << "Iteration:" << counter 
                          << " Poly:" << gen.Poly() 
                          << " IPS:" << (counter / now)
                          << " Error:" << gen.Error()
                          << " ET:" << t_elapsed
                          << " Generation:" << gen.Generation() << "\n";
                std::ostringstream os;

                os << "Gen:" << gen.Generation() << " IPS:" << (counter / now) << " Err:" << gen.Error();
                SDL_WM_SetCaption(os.str().c_str(), NULL);
            }

            /* Restart */
            gettimeofday(&t_start, NULL);
        }
    }
}

