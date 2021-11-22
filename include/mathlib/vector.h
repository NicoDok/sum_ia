#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <initializer_list>

#include "functions.h"

namespace MathLib {

template <class T>
class vector {
    public:
        // Constr/Destr
        vector(uint64_t size);
        vector(std::initializer_list<T>);
        ~vector() {}

        // Methods
        uint64_t size() { return m_size; }
        uint64_t capacity() { return m_capacity; }
        void resize(uint64_t new_size);
        void reserve(uint64_t new_capacity);

        // Operators
        T& operator [](uint64_t index);
        const T& operator [](uint64_t index) const;

    private:
        uint64_t m_size;
        uint64_t m_capacity;
        std::unique_ptr<T[]> m_data;
};

template <class T>
vector<T>::vector(uint64_t size) : m_size(size), m_capacity(next_pow2(size)) {
    m_data = std::unique_ptr<T[]>(new T[m_capacity]);
}

template<class T>
vector<T>::vector(std::initializer_list<T> s) {
    reserve(s.size());  // get the right amount of space
    std::uninitialized_copy(s.begin(), s.end(), m_data.get());   // initialize elements (in elem[0:s.size()))
    m_size = s.size();  // set vector size
}

template <class T>
void vector<T>::resize(uint64_t new_size) {
    if (new_size > m_capacity) {
        m_capacity = next_pow2(new_size);
        m_data =  std::unique_ptr<T[]>(new T[m_capacity]);

        auto new_data = new T[m_capacity];
        std::copy(m_data.get(), m_data.get() + m_size, new_data);
        m_data.release();
        m_data = std::unique_ptr<T[]>(new_data);
    }
    m_size = new_size;
}

template <class T>
void vector<T>::reserve(uint64_t new_capacity) {
    if (new_capacity > m_capacity) {
        m_capacity = new_capacity;
        m_data = std::unique_ptr<T[]>(new T[m_capacity]);

        auto new_data = new T[m_capacity];
        std::copy(m_data.get(), m_data.get() + m_size, new_data);
        m_data.release();
        m_data = std::unique_ptr<T[]>(new_data);
    }
}

template <class T>
T& vector<T>::operator [](uint64_t index) 
{ 
    return (index < m_size) ? m_data[index] : throw std::out_of_range("index out of bound"); 
}

template <class T>
const T& vector<T>::operator [](uint64_t index) const 
{ 
    return vector<T>::operator[](index);
}


}