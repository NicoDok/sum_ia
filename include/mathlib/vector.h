#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <initializer_list>

#include "functions.h"

namespace MathLib {

template <class value_type, class size_type = uint64_t> class vector;

template <class value_type, class size_type>
class vector {
    public:
        // Constr
        vector();
        explicit vector(size_type size);
        vector(size_type n, const value_type& val);

        vector(const vector& x);
        vector(vector&& x);

        template <class InputIterator>
        vector(InputIterator first, InputIterator last);

        vector(std::initializer_list<value_type>);

        // Destr
        ~vector() {}

        // Methods
        size_type size() { return m_size; }
        size_type max_size() { return std::numeric_limits<size_type>::max()-1; }
        size_type capacity() { return m_capacity; }
        void resize(size_type new_size);
        void reserve(size_type new_capacity);

        // Operators
        value_type& operator [](size_type index);
        const value_type& operator [](size_type index) const;

    private:
        size_type m_size;
        size_type m_capacity;
        std::unique_ptr<value_type[]> m_data;
};

template <class value_type, class size_type>
vector<value_type, size_type>::vector(): m_size(0), m_capacity(0), m_data(std::unique_ptr<value_type[]>()) {
}

template <class value_type, class size_type>
vector<value_type, size_type>::vector(size_type size): m_size(size), m_capacity(next_pow2(size)) {
    m_data = std::unique_ptr<value_type[]>(new value_type[m_capacity]);
}

template <class value_type, class size_type>
vector<value_type, size_type>::vector(size_type n, const value_type& val): m_size(size), m_capacity(next_pow2(size)) {
    m_data = std::unique_ptr<value_type[]>(new value_type[m_capacity](val));

}

template <class value_type, class size_type>
vector<value_type, size_type>::vector(const vector& x): m_size(x.m_size), m_capacity(x.m_capacity) {
    auto new_data = new value_type[m_capacity];
    std::copy(x.m_data.get(), x.m_data.get() + m_size, new_data);
    m_data = std::unique_ptr<value_type[]>(new_data);
}

template <class value_type, class size_type>
vector<value_type, size_type>::vector(vector&& x): m_size(x.m_size), m_capacity(x.m_capacity) {
    m_data = std::move(x.m_data);
}

template <class value_type, class size_type>
template <class InputIterator>
vector<value_type, size_type>::vector(InputIterator first, InputIterator last): m_size(std::distance((first, last))), m_capacity(next_pow2(size)) {
    reserve(m_capacity);
    std::uninitialized_copy(*first, *last, m_data.get());
}

template <class value_type, class size_type>
vector<value_type, size_type>::vector(std::initializer_list<value_type> s): vector() {
    reserve(s.size());
    std::uninitialized_copy(s.begin(), s.end(), m_data.get());
    m_size = s.size();
}

template <class value_type, class size_type>
void vector<value_type, size_type>::resize(size_type new_size) {
    if (new_size > m_capacity) {
        m_capacity = next_pow2(new_size);
        m_data =  std::unique_ptr<value_type[]>(new value_type[m_capacity]);

        auto new_data = new value_type[m_capacity];
        std::copy(m_data.get(), m_data.get() + m_size, new_data);
        m_data.reset();
        m_data = std::unique_ptr<value_type[]>(new_data);
    }
    m_size = new_size;
}

template <class value_type, class size_type>
void vector<value_type, size_type>::reserve(size_type new_capacity) {
    if (new_capacity > m_capacity) {
        m_capacity = new_capacity;
        m_data = std::unique_ptr<value_type[]>(new value_type[m_capacity]);

        auto new_data = new value_type[m_capacity];
        std::copy(m_data.get(), m_data.get() + m_size, new_data);
        m_data.reset();
        m_data = std::unique_ptr<value_type[]>(new_data);
    }
}

template <class value_type, class size_type>
value_type& vector<value_type, size_type>::operator [](size_type index) 
{ 
    return (index < m_size) ? m_data[index] : throw std::out_of_range("index out of bound"); 
}

template <class value_type, class size_type>
const value_type& vector<value_type, size_type>::operator [](size_type index) const 
{ 
    return vector<value_type, size_type>::operator[](index);
}

}