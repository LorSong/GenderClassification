# GenderClassification

Репозиторий для размещения тестового задания от NtechLab.  
Основная работа - создание модели, которая определяет пол человека по изображению его лица.

### Описание файлов
<details>
  <summary>Раскрыть</summary>  
  
  1. MaxSubArray.py - содержит функцию findMaxSubArray(A) к первому заданию.
  2. GenderClassification_#.ipynb - Jupyter notebooks с шагами по обучению сети
  3. process.py - cкрипт для использования нейросети
  4. model - папка с tf.model, которую использует скрипт process.py для загрузки модели
</details>

## Описание модели
<details>
  <summary>Раскрыть</summary>  
  
  В работе
</details>

## Инструкция по применению сети.
<details>
  <summary>Раскрыть</summary>  
  
  1) Убедитесь, что у вас установлен python с tensorflow версии 2 и выше
  2) Скопируйте файл process.py вместе с папкой model в одну директорию.

  ![](desc_images/folder_files.png)

  3) Запустите командную строку и перейдите в директорию с файлами. Можете разместить изображения в эту же папку.

  ![](desc_images/changefolder.jpg)

  4) Запустите скрипт указав путь к папке с изображениями.

  ![](desc_images/process_exec.png)
</details>
