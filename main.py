from abc import ABC, abstractmethod
import math
import numpy as np
import matplotlib.pyplot as plt


## @package main


## Абстрактный класс RandomVariable.
#  Интерфейс для любой случайной величины.
class RandomVariable(ABC):
  ## Абстрактный метод pdf.
  #  Вычисление плотности вероятности.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  @abstractmethod
  def pdf(self, x):
    pass

  ## Абстрактный метод quantile.
  #  Вычисление квантиля уровня alpha.
  #  @param self Указатель на объект.
  #  @param alpha Уровень квантеля.
  @abstractmethod
  def quantile(self, alpha):
    pass

  ## Абстрактный метод cdf.
  #  Вычисление интегральной функции распределения.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  @abstractmethod
  def cdf(self, x):
    pass


  ## @var x
  #  Значение случайной величины.

  ## @var alpha
  #  Уровень квантеля.


## Определяем класс NormalRandomVariable(RandomVariable).
#  Класс, вычисляющий нормальные случайные величины.
class NormalRandomVariable(RandomVariable):
  ##  Инициализируем атрибуты объекта класса.
  #  @param self Указатель на объект.
  #  @param location Параметр сдвига.
  #  @param scale Параметр масштаба.
  def __init__(self, location=0, scale=1) -> None:
    super().__init__()
    self.location = location
    self.scale = scale

  ##  Вычисление интегральной функции распределния для нормальной случайно величины.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  def cdf(self, x):
    z = (x - self.location) / self.scale
    if z <= 0:
      return 0.852 * math.exp(-math.pow((-z + 1.5774) / 2.0637, 2.34))
    return 1 - 0.852 * math.exp(-math.pow((z + 1.5774) / 2.0637, 2.34))

  ## Вычисление плотности вероятности для нормальной случайно величины.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  def pdf(self, x):
    z = (x - self.location) / self.scale
    return math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * self.scale)

  ##  Вычисление квантиля уровня alpha для нормальной случайно величины.
  #  @param self Указатель на объект.
  #  @param alpha Уровень квантеля.
  def quantile(self, alpha):
    return self.location + 4.91 * self.scale * (math.pow(alpha, 0.14) - math.pow(1 - alpha, 0.14))

  ## @var x
  #  Значение случайной величины.

  ## @var alpha
  #  Уровень квантеля.

  ## @var location
  #  Параметр сдвига.

  ## @var scale
  #  Параметр масштаба.


## Абстрактный класс RandomNumberGenerator.
#  Интерфейс для генератора псевдослучайных величин.
class RandomNumberGenerator(ABC):
  ##  Инициализируем атрибуты объекта класса.
  #  @param self Указатель на объект.
  #  @param random_variable Псевдослучайные величины.
  def __init__(self, random_variable: RandomVariable):
    self.random_variable = random_variable

  ## Абстрактный метод get.
  #  Возвращает выборку объема N.
  #  @param self Указатель на объект.
  #  @param N Объём выборки.
  @abstractmethod
  def get(self, N):
    pass

  ## @var random_variable
  #  Псевдослучайные величины.

  ## @var N
  #  Объём выборки.


## Определяем класс SimpleRandomNumberGenerator(RandomNumberGenerator).
#  Генератор псевдослучайных величин, который получает выборку, подчиняющий требуемому распределению.
class SimpleRandomNumberGenerator(RandomNumberGenerator):
  ##  Инициализируем атрибуты объекта класса.
  #  @param self Указатель на объект.
  #  @param random_variable Псевдослучайные величины.
  def __init__(self, random_variable: RandomVariable):
    super().__init__(random_variable)

  ##  Возвращает выборку объема N.
  #  @param self Указатель на объект.
  #  @param N Объём выборки.
  def get(self, N):
    us = np.random.uniform(0, 1, N)
    return np.vectorize(self.random_variable.quantile)(us)

  ## @var random_variable
  #  Псевдослучайные величины.

  ## @var N
  #  Объём выборки.


##  Функциуя для рисования графиков.
#  @param xs Ось x (значения случайной величины).
#  @param ys Ось y (значения вероятностей).
#  @param colors Цвет функции.
def plot(xs, ys, colors):
  for x, y, c in zip(xs, ys, colors):
    plt.plot(x, y, c)
  plt.show()

  ## @var xs
  #  Ось x (значения случайной величины).

  ## @var ys
  #  Ось y (значения вероятностей).

  ## @var colors
  #  Цвет функции.


## Абстрактный класс Estimation.
#  Класс для всех оценок.
class Estimation(ABC):
  ##  Инициализируем атрибуты объекта класса.
  #  @param self Указатель на объект.
  #  @param sample Выборка данных
  def __init__(self, sample):
    self.sample = sample

  ## @var sample
  #  Выборка данных.


## Определяем класс EDF(Estimation).
#  Класс вычисляет эмпирическую функцию распределения.
class EDF(Estimation):
  ##  Вычисление значения функции Хевисайда.
  #  @param x Значение случайной величины.
  def heaviside_function(x):
    if x > 0:
      return 1
    else:
      return 0

  ##  Вычисление значения эмпирической функции распределения.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  def value(self, x):
    return np.mean(np.vectorize(EDF.heaviside_function)(x - self.sample))

  ## @var x
  #  Значение случайной величины.


## Определяем класс SmoothedRandomVariable(RandomVariable, Estimation).
# Класс вычисляет непараметрическую случайную величину
class SmoothedRandomVariable(RandomVariable, Estimation):
  ##  Вычисление значения ядерной функции.
  #  @param x Значение случайной величины.
  def _k(x):
    if abs(x) <= 1:
      return 0.75 * (1 - x * x)
    else:
      return 0

  ##  Вычисление значения сглаженной эмпирической оценки.
  #  @param x Значение случайной величины.
  def _K(x):
    if x < -1:
      return 0
    elif -1 <= x < 1:
      return 0.5 + 0.75 * (x - x ** 3 / 3)
    else:
      return 1

  ##  Инициализируем атрибуты объекта класса.
  #  @param self Указатель на объект.
  #  @param sample Выборка данных
  #  @param h Параметр размытости
  def __init__(self, sample, h):
    super().__init__(sample)
    self.h = h

  ##  Вычисление значения оценки Розенблатта-Парзена.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  def pdf(self, x):
    return np.mean([SmoothedRandomVariable._k((x - y) / self.h) for y in self.sample]) / self.h

  ##  Вычисление значения сглаженной эмпирической оценки.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  def cdf(self, x):
    return np.mean([SmoothedRandomVariable._K((x - y) / self.h) for y in self.sample])

  ##  Вычисление квантиля уровня alpha.
  #  @param self Указатель на объект.
  #  @param alpha Уровень квантиля.
  def quantile(self, alpha):
    raise NotImplementedError

  ## @var x
  #  Значение случайной величины.

  ## @var h
  #  Параметр размытости.

  ## @var sample
  #  Выборка данных.

  ## @var alpha
  #  Уровень квантиля.


## Определяем класс Histogram(Estimation).
# Класс, вычисляющий гистограмму.
class Histogram(Estimation):
  ## Определяем вложенный класс Interval.
  # Класс вычисляет интервалы для гистограммы.
  class Interval:
    ##  Инициализируем атрибуты объекта класса.
    #  @param self Указатель на объект.
    #  @param a Левая граница
    #  @param b Правая граница
    def __init__(self, a, b):
      self.a = a
      self.b = b

    ##  Проверяем попадание x в интервал.
    #  @param self Указатель на объект.
    #  @param x Значение случайной величины.
    def is_in(self, x):
      return x >= self.a and x <= self.b

    ##  Возвращает строковые представления интервала.
    #  @param self Указатель на объект.
    def __repr__(self):
      return f'({self.a}, {self.b})'

      ## @var a
      #  Левая граница

      ## @var b
      #  Правая граница

      ## @var x
      #  Значение случайной величины.

  ##  Инициализируем атрибуты объекта класса.
  #  @param self Указатель на объект.
  #  @param sample Выборка
  #  @param m Количество интервалов
  def __init__(self, sample, m):
    super().__init__(sample)
    self.m = m

    self.init_intervals()

  ##  Определяет интервалы.
  #  @param self Указатель на объект.
  def init_intervals(self):
    left_boundary_of_intervals = np.linspace(np.min(sample), np.max(sample), self.m + 1)[:-1]
    right_boundary_of_intervals = np.concatenate((left_boundary_of_intervals[1:], [np.max(sample)]))

    self.intervals = [ Histogram.Interval(a, b) for a,b in zip(left_boundary_of_intervals, right_boundary_of_intervals)]

    self.sub_interval_width = right_boundary_of_intervals[0] - left_boundary_of_intervals[0]

  ##  Определяет интервалы.
  #  @param self Указатель на объект.
  def get_interval(self, x):
    for i in self.intervals:
      if i.is_in(x):
        return i
    return None

  ##  Распределение значений по интервалам.
  #  @param self Указатель на объект.
  #  @param interval Интервал.
  def get_sample_by_interval(self, interval):
    return np.array(list(filter(lambda x: interval.is_in(x), self.sample)))

  ##  Вычисление значения оценки плотности вероятности в точке x.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  def value(self, x):
    return len(self.get_sample_by_interval(self.get_interval(x))) / ( self.sub_interval_width * len(self.sample) )


  ## @var x
  #  Значение случайной величины.

  ## @var sample
  #  Выборка

  ## @var m
  #  Количество интервалов

  ## @var interval
  #  interval Интервал.


## Определяем класс UniformRandomVariable(RandomVariable).
# Класс для вычисления равномерной функции распределения.
class UniformRandomVariable(RandomVariable):
  ##  Инициализируем атрибуты объекта класса.
  #  @param self Указатель на объект.
  #  @param location Параметр сдвига
  #  @param scale Параметр масштаба
  def __init__(self, location=0, scale=1):
    super().__init__()
    self.location = location
    self.scale = scale

  ## Вычисление плотности вероятности для равномерной функции распределения.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  def pdf(self, x):
    if self.location <= x <= self.scale:
      return 1 / (self.scale - self.location)
    else:
      return 0

  ##  Вычисление интегральной функции распределния для равномерной функции распределения.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  def cdf(self, x):
    if x <= self.location:
      return 0
    elif x >= self.scale:
      return 1
    else:
      return (x - self.location) / (self.scale - self.location)

  ##  Вычисление квантиля уровня alpha для равномерной функции распределения.
  #  @param self Указатель на объект.
  #  @param alpha Уровень квантеля.
  def quantile(self, alpha):
    return self.location + alpha * (self.scale - self.location)

  ## @var location
  #  Параметр сдвига

  ## @var scale
  #  Параметр масштаба

  ## @var x
  #  Значение случайной величины.

  ## @var alpha
  #  Уровень квантеля.


## Определяем класс LaplaceRandomVariable(RandomVariable).
# Класс для вычисления функции распределения Лапласа.
class LaplaceRandomVariable(RandomVariable):
  ##  Инициализируем атрибуты объекта класса.
  #  @param self Указатель на объект.
  #  @param location Параметр сдвига
  #  @param scale Параметр масштаба
  def __init__(self, location=0, scale=1):
    self.location = location
    self.scale = scale

  ## Вычисление плотности вероятности для функции распределения Лапласа.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  def pdf(self, x):
    return 0.5 * self.scale * math.exp(-self.scale * abs(x - self.location))

  ##  Вычисление интегральной функции распределния для функции распределения Лапласа.
  #  @param self Указатель на объект.
  #  @param x Значение случайной величины.
  def cdf(self, x):
    if x < self.location:
      return 0.5 * math.exp((x - self.location) / self.scale)
    else:
      return 1 - 0.5 * math.exp(-(x - self.location) / self.scale)

  ##  Вечисление квантиля уровня alpha для функции распределения Лапласа.
  #  @param self Указатель на объект.
  #  @param alpha Уровень квантеля.
  def quantile(self, alpha):
    if alpha == 0.5:
      return self.location
    elif alpha < 0.5:
      return self.location - self.scale * math.log1p(-2 * alpha)
    else:
      return self.location + self.scale * math.log1p(2 * alpha - 1)

      ## @var location
      #  Параметр сдвига

      ## @var scale
      #  Параметр масштаба

      ## @var x
      #  Значение случайной величины.

      ## @var alpha
      #  Уровень квантеля.



