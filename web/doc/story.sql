
-- 训练
drop table IF EXISTS `db_ai_story`.`t_local_train`;
create table `db_ai_story`.`t_local_train`(
  `voice_id` varchar(255) NOT NULL,
  `task_id` varchar(255) NOT NULL,
  `audio_file` varchar(512) DEFAULT NULL,       -- 输入
  `create_time` DATETIME NOT NULL DEFAULT NOW(),
  PRIMARY KEY (`voice_id`)
)ENGINE=InnoDB DEFAULT CHARSET=UTF8MB4;


-- 推理
drop table IF EXISTS `db_ai_story`.`t_local_infer`;
create table `db_ai_story`.`t_local_infer`(
  `task_id` varchar(255) NOT NULL,
  `voice_id` varchar(255) NOT NULL,
  `audio_file` varchar(512) DEFAULT NULL,       -- 输出
  `create_time` DATETIME NOT NULL DEFAULT NOW(),
  PRIMARY KEY (`task_id`)
)ENGINE=InnoDB DEFAULT CHARSET=UTF8MB4;