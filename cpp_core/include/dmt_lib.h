#ifndef DMT_LIB_H
#define DMT_LIB_H

#ifdef __cplusplus
extern "C" {
#endif

// Экспорт функций для Windows DLL
#ifdef _WIN32
    #ifdef DMT_LIB_EXPORTS
        #define DMT_API __declspec(dllexport)
    #else
        #define DMT_API __declspec(dllimport)
    #endif
#else
    #define DMT_API
#endif

// Непрозрачный указатель на модель
typedef void* DMT_Handle;

// Создание модели
// depth_memory: глубина памяти (обычно 7)
// num_features: количество входных признаков
// layers: массив размеров слоев
// num_layers: количество слоев
DMT_API DMT_Handle DMT_Create(int depth_memory, int num_features, const int* layers, int num_layers);

// Удаление модели
DMT_API void DMT_Destroy(DMT_Handle handle);

// Обучение модели
// X: массив массивов входных данных (X[i*num_features + j] = j-й признак i-го примера)
// y: массив меток
// num_samples: количество примеров
// epochs: количество эпох
DMT_API void DMT_LearnSL(DMT_Handle handle, const int* X, const int* y, int num_samples, int epochs);

// Обучение с подкреплением (один пример)
DMT_API void DMT_LearnRL(DMT_Handle handle, const int* X_input, int reward);

// Предсказание
// X_input: входной вектор (размер num_features)
// y_pred: выходной указатель для предсказания
// Returns: 0 при успехе, -1 при ошибке
DMT_API int DMT_Predict(DMT_Handle handle, const int* X_input, int* y_pred);

// Сохранение весов в бинарный файл
// filename: путь к файлу
// Returns: 0 при успехе, -1 при ошибке
DMT_API int DMT_SaveWeights(DMT_Handle handle, const char* filename);

// Загрузка весов из бинарного файла
// filename: путь к файлу
// Returns: 0 при успехе, -1 при ошибке
DMT_API int DMT_LoadWeights(DMT_Handle handle, const char* filename);

// Получение информации о модели
DMT_API int DMT_GetNumFeatures(DMT_Handle handle);
DMT_API int DMT_GetNumLayers(DMT_Handle handle);
DMT_API int DMT_GetLayerSize(DMT_Handle handle, int layer_idx);
DMT_API int DMT_GetDepthMemory(DMT_Handle handle);

#ifdef __cplusplus
}
#endif

#endif // DMT_LIB_H

