#ifndef POINT_H
#define POINT_H

#include "serializer.h"

template <typename T>
class Point : public Serializable
{
    T x_, y_;
public:    
    explicit Point(T x = 0, T y = 0) :
        x_(x), y_(y) {}

    Point operator+(const Point &rhs) const {
        return Point(x_ + rhs.x_, y_ + rhs.y_);
    }
    Point &operator+=(const Point &rhs) {
        x_ += rhs.x_;
        y_ += rhs.y_;
        return *this;
    }
    Point operator-(const Point &rhs) const {
        return Point(x_ - rhs.x_, y_ - rhs.y_);
    }
    Point &operator-=(const Point &rhs) {
        x_ -= rhs.x_;
        y_ -= rhs.y_;
        return *this;
    }
    Point operator*(float f) const {
        return Point(x_ * f, y_ * f);
    }
    Point &operator*=(float f) {
        x_ *= f;
        y_ *= f;
        return *this;
    }

    Point operator/(float f) const {
        return Point(x_ / f, y_ / f);
    }
    Point &operator/=(float f) {
        x_ /= f;
        y_ /= f;
        return *this;
    }

    T X() const { return x_; }
    T Y() const { return y_; }

    void X(T x) { x_ = x; }
    void Y(T y) { y_ = y; }

    bool operator==(const Point &rhs) const {
        return (x_ == rhs.x_ && y_ == rhs.y_);
    }
    bool operator!=(const Point &rhs) const {
        return (x_ != rhs.x_ || y_ != rhs.y_);
    }

    Point &operator=(const Point &rhs) {
        x_ = rhs.x_;
        y_ = rhs.y_;
        return *this;
    }

    void serialize(Serializer &s) {
        s & x_;
        s & y_;
    }
};

#endif
