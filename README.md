# GenderClassification

Репозиторий для размещения тестового задания от NtechLab.  
Основная работа - создание модели, которая определяет пол человека по изображению его лица.

## Описание файлов
<details>
  <summary>Раскрыть</summary><br/>
  
  1. MaxSubArray.py - содержит функцию findMaxSubArray(A) к первому заданию.
  2. GenderClassification_#.ipynb - Jupyter notebooks с шагами по обучению сети
  3. process.py - cкрипт для использования нейросети (смотри инструкцию ниже)
  4. model - папка с tf.model, которую использует скрипт process.py для загрузки модели
</details>
  5. train.py - скрипт для обучения нейросети. Создает папку model. (смотри инструкцию ниже). 
  6. Gender_Clf_Utilities.py - дополнительные функции, которые были созданы во экспериментов с моделями. Используются в Jupyter notebooks

## Описание модели
<details>
  <summary>Раскрыть</summary><br/>  
  
  В работе
</details>

## Инструкция по применению сети.
<details>
  <summary>Раскрыть</summary><br/> 
  
  1) Убедитесь, что у вас установлен python с tensorflow версии 2 и выше
  2) Скопируйте файл process.py вместе с папкой model в одну директорию. Можете разместить изображения в эту же папку.

  ![](desc_images/folder_files.png)

  3) Запустите командную строку и перейдите в директорию с файлами. 

  ![](desc_images/changefolder.jpg)

  4) Запустите скрипт указав путь к папке с изображениями.

  ![](desc_images/process_exec.png)

  5) После выполения, в папке где находится скрипт, появится новый файл process_results.json. В нем будут размещены результаты
  в виде { ‘img_1.jpg’: ‘male’, ‘img_2.jpg’: ‘female’, ...}
  </details>
